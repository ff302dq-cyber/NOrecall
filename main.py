"""
消息撤回同步 (RecallSync)

群聊：群友撤回了“对 Bot 说的话”（@ 了 Bot、带唤醒词、或走了指令）时
私聊：用户撤回任何一条消息时（私聊里每句话都是对 Bot 说的）
  1. 发一条“系统报错”（Bot 还没回复时报不报由 error_when_no_reply 决定）（fold_window 秒内的多次撤回折叠成一条，设 0 则每次单独报错）
  2. Bot 还没回复 -> 直接不回（还没调模型的话连钱都不花）
  3. Bot 已经回复 -> 把 Bot 自己那几条也撤掉（含分段回复）
兼容消息合并插件：合并后的回复会关联到这一批里的每条原消息，撤其中任意一条都算。
群里普通聊天消息的撤回不受影响；开了主动回复的 Bot 只撤自己的回复，不报错。

依赖：OneBot v11 实现端（NapCat / SnowLuma 等）支持 group_recall / friend_recall 通知、delete_msg、
get_group_msg_history / get_friend_msg_history（NapCat 均支持）。
"""

import asyncio
import random
import sys
import time
import types

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

try:
    from astrbot.api.message_components import At
except Exception:  # pragma: no cover
    At = None

# 仿 AstrBot 真实报错：“LLM 响应错误: {detail}”，detail 仿上游 OpenAI 兼容接口的报错格式
DEFAULT_TEMPLATES = [
    "LLM 响应错误: All chat models failed: BadRequestError: Error code: 400 - "
    "{{'error': {{'message': 'Source message was withdrawn by sender. "
    "(blockReason: SOURCE_WITHDRAWN)', 'type': 'upstream_error', 'param': '', 'code': 400}}}}",
    "LLM 响应错误: All chat models failed: BadRequestError: Error code: 400 - "
    "{{'error': {{'message': 'Context message not found or has been revoked. "
    "(request_id: {trace})', 'type': 'invalid_request_error', 'param': 'messages', 'code': 400}}}}",
    "LLM 响应错误: All chat models failed: NotFoundError: Error code: 404 - "
    "{{'error': {{'message': 'Referenced message is no longer available. "
    "(blockReason: MESSAGE_RECALLED)', 'type': 'upstream_error', 'param': '', 'code': 404}}}}",
]

KEEP_SECONDS = 300  # 记录保留 5 分钟，QQ 撤回窗口只有 2 分钟，够用

GROUP, PRIVATE = "group", "private"


@register("astrbot_plugin_recall_sync", "junjun", "消息撤回同步", "1.7.0")
class RecallSync(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.cfg = config or {}
        self.clients = {}      # self_id -> OneBot client
        self.woken = {}        # key -> 时间  对 Bot 说的消息
        self.pending = {}      # key -> {kind, tid, self_id, t_start, t_end, ids, revoked}
        self.recalled = {}     # key -> 撤回时间
        self.errored = set()   # 已经为它报过错的消息
        self.claimed = set()   # 已认领的 Bot 消息 id，防止两条回复抢同一条
        self.error_ids = set() # 伪报错自己的消息 id
        self.fold = {}         # "self_id:kind:tid" -> {"n": 次数}
        # 给自己直接发消息、绕开 AstrBot 发送流程的插件（如六爻）用的登记口
        sys.modules["astrbot_recall_sync"] = types.SimpleNamespace(
            is_recalled=self.is_recalled, register_reply=self.register_reply
        )

    # ---------- 对外接口 ----------

    def is_recalled(self, event: AstrMessageEvent) -> bool:
        """触发这条事件的原消息是不是已经被撤回了。发结果前问一句，撤了就别发。"""
        try:
            if not self._on(event):
                return False
            return any(k in self.recalled for k in self._source_keys(event))
        except Exception:
            return False

    async def register_reply(self, event: AstrMessageEvent, message_ids=None, t_start=None):
        """登记“这几条 Bot 消息是回复 event 的”。message_ids 拿不到时，
        按 t_start 到现在这段时间去群/私聊历史里找 Bot 自己发的消息。"""
        if not self._on(event):
            return
        self._remember_client(event)
        self._mark_woken(event)
        keys = self._source_keys(event)
        if not keys:
            return
        now = time.time()
        p = next((self.pending[k] for k in keys if k in self.pending), None)
        if p is None:
            kind, tid = self._target(event)
            p = {
                "kind": kind,
                "tid": tid,
                "self_id": str(event.get_self_id()),
                "t_start": t_start or now,
                "t_end": now,
                "ids": [],
                "revoked": False,
            }
        else:
            p["t_end"] = now
        for k in keys:
            self.pending[k] = p

        ids = [str(i) for i in (message_ids or []) if i]
        for i in ids:
            self.claimed.add(i)
        if not ids:
            await asyncio.sleep(float(self._c("lookup_delay", 1.0)))
            win = dict(p, t_start=t_start or p["t_start"], t_end=now)
            ids = await self._find_bot_msgs(win)
        if p["revoked"]:
            # 原消息早就撤了，后补登记的这几条直接撤掉
            for i in ids:
                await self._call(p["self_id"], "delete_msg", message_id=int(i))
            return
        p["ids"].extend(i for i in ids if i not in p["ids"])
        hit = next((k for k in keys if k in self.recalled), None)
        if hit:
            await self._revoke(hit, p)

    # ---------- 小工具 ----------

    def _c(self, key, default):
        try:
            v = self.cfg.get(key, default)
        except Exception:
            return default
        return default if v is None else v

    @staticmethod
    def _target(event: AstrMessageEvent):
        """返回 (会话类型, 群号或对方QQ号)。"""
        gid = event.get_group_id()
        if gid:
            return GROUP, str(gid)
        return PRIVATE, str(event.get_sender_id())

    def _on(self, event: AstrMessageEvent) -> bool:
        try:
            if not self._c("enable", True) or event.get_platform_name() != "aiocqhttp":
                return False
            if event.get_group_id():
                return True
            return bool(self._c("private_enable", True))
        except Exception:
            return False

    @staticmethod
    def _key(self_id, mid) -> str:
        return f"{self_id}:{mid}"

    @staticmethod
    def _is_wake(event: AstrMessageEvent) -> bool:
        """这条消息是不是冲 Bot 说的：私聊全算；群里看 @Bot / 唤醒词 / 指令。"""
        if not event.get_group_id():
            return True
        if getattr(event, "is_at_or_wake_command", False):
            return True
        if At is not None:
            try:
                self_id = str(event.get_self_id())
                for comp in event.get_messages():
                    if isinstance(comp, At) and str(getattr(comp, "qq", "")) == self_id:
                        return True
            except Exception:
                pass
        return False

    @staticmethod
    def _source_ids(event: AstrMessageEvent) -> list:
        """这条事件对应的原始消息 id。合并插件造的新事件 id 为空，原消息 id 放在 extra 里。"""
        ids = []
        try:
            extra = event.get_extra("combined_message_ids")
        except Exception:
            extra = None
        if isinstance(extra, (list, tuple)):
            ids = [str(i) for i in extra if i]
        if not ids:
            mid = str(getattr(event.message_obj, "message_id", "") or "")
            if mid:
                ids = [mid]
        return ids

    def _source_keys(self, event: AstrMessageEvent) -> list:
        self_id = event.get_self_id()
        return [self._key(self_id, i) for i in self._source_ids(event)]

    def _remember_client(self, event: AstrMessageEvent):
        bot = getattr(event, "bot", None)
        if bot is not None:
            self.clients[str(event.get_self_id())] = bot

    def _mark_woken(self, event: AstrMessageEvent):
        if self._is_wake(event):
            now = time.time()
            for key in self._source_keys(event):
                self.woken.setdefault(key, now)

    def _gc(self):
        now = time.time()
        for k in [k for k, v in self.pending.items() if now - v["t_start"] > KEEP_SECONDS]:
            self.pending.pop(k, None)
        for d in (self.woken, self.recalled):
            for k in [k for k, t in d.items() if now - t > KEEP_SECONDS]:
                d.pop(k, None)
        if len(self.errored) > 3000:
            self.errored.clear()
        if len(self.claimed) > 3000:
            self.claimed.clear()
        if len(self.error_ids) > 500:
            self.error_ids.clear()

    async def _call(self, self_id, action, **params):
        client = self.clients.get(str(self_id))
        if client is None:
            return None
        try:
            return await client.api.call_action(action, **params)
        except Exception as e:
            logger.warning(f"[RecallSync] {action} failed: {e}")
            return None

    def _error_once(self, key, self_id, kind, tid):
        """同一条被撤回的消息只报一次错。"""
        if key in self.errored:
            return
        self.errored.add(key)
        self._fold_error(self_id, kind, tid)

    # ---------- 1. 所有消息：记下“对 Bot 说的话”；撤回通知：处理 ----------

    # 优先级调高：必须抢在消息合并插件（会 stop_event 截走原消息）之前登记
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100000)
    async def on_raw(self, event: AstrMessageEvent):
        raw = getattr(event.message_obj, "raw_message", None)
        notice = raw.get("notice_type") if isinstance(raw, dict) and raw.get("post_type") == "notice" else None

        if notice not in ("group_recall", "friend_recall"):
            if self._on(event):
                self._remember_client(event)
                self._mark_woken(event)
            return

        event.stop_event()
        if not self._c("enable", True):
            return
        if notice == "friend_recall" and not self._c("private_enable", True):
            return

        self._remember_client(event)
        self_id = str(raw.get("self_id") or event.get_self_id())
        user_id = str(raw.get("user_id"))
        if user_id == self_id:
            return  # Bot 自己的消息被撤，不管

        if notice == "group_recall":
            kind, tid = GROUP, str(raw.get("group_id"))
            operator = str(raw.get("operator_id") or user_id)
        else:
            kind, tid = PRIVATE, user_id
            operator = user_id

        mid = str(raw.get("message_id"))
        key = self._key(self_id, mid)
        if self._c("debug", False):
            logger.info(
                f"[RecallSync] recall notice: {kind}={tid} msg={mid} "
                f"woken={key in self.woken} replied={key in self.pending}"
            )

        self._gc()
        self.recalled[key] = time.time()

        # 报错：撤回的是对 Bot 说的话，且是本人撤回（管理员撤别人的默认不报）
        # Bot 还没发出回复时是否报错，由 error_when_no_reply 决定
        replied = key in self.pending
        if key in self.woken and (replied or self._c("error_when_no_reply", True)):
            if operator == user_id or not self._c("ignore_admin_recall", True):
                self._error_once(key, self_id, kind, tid)

        # 双撤：Bot 已经发出回复的才撤
        p = self.pending.get(key)
        if p and p["t_end"] is not None:
            await self._revoke(key, p)
        # 还在生成或发送中的，交给下面的钩子拦截

    # ---------- 自检：看协议端支不支持需要的接口 ----------

    @filter.command("rs_check")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def rs_check(self, event: AstrMessageEvent, group_id: str = ""):
        """管理员私聊发 /rs_check [群号]，检查查历史接口能不能用"""
        self._remember_client(event)
        self_id = str(event.get_self_id())
        lines = [f"bot={self_id}"]

        ret = await self._call(
            self_id, "get_friend_msg_history", user_id=int(event.get_sender_id()), count=10
        )
        lines.append(self._check_line("friend_history", ret, self_id))

        gid = group_id or event.get_group_id()
        if gid:
            ret = await self._call(
                self_id, "get_group_msg_history", group_id=int(gid), count=10
            )
            lines.append(self._check_line(f"group_history({gid})", ret, self_id))
        else:
            lines.append("group_history: 未测试（/rs_check 后面带个群号）")

        lines.append("撤回通知：开 debug 后撤一条消息，看后台有没有 recall notice")
        yield event.plain_result("\n".join(lines))

    @staticmethod
    def _check_line(name, ret, self_id):
        if not isinstance(ret, dict):
            return f"{name}: 不可用"
        msgs = ret.get("messages", [])
        mine = sum(
            1
            for m in msgs
            if str((m.get("sender") or {}).get("user_id", m.get("user_id", ""))) == self_id
        )
        tip = "OK" if mine else "能调用，但没找到 Bot 自己的消息，双撤可能失效"
        return f"{name}: {tip}（{len(msgs)} 条，其中 Bot 自己 {mine} 条）"

    # ---------- 2. 调模型前：已撤回就不调 ----------

    # 优先级调高：赶在 bill 扣费之前拦下已撤回的消息，免得先挂单再等 10 分钟退回
    @filter.on_llm_request(priority=100000)
    async def before_llm(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._on(event):
            return
        self._remember_client(event)
        self._mark_woken(event)
        delay = float(self._c("pre_delay_max", 0) or 0)
        if delay > 0:
            await asyncio.sleep(random.uniform(delay * 0.4, delay))
        if any(k in self.recalled for k in self._source_keys(event)):
            event.stop_event()

    # ---------- 3. 发送前：已撤回就不发，否则登记 ----------

    @filter.on_decorating_result()
    async def before_send(self, event: AstrMessageEvent):
        if not self._on(event):
            return
        result = event.get_result()
        if not result or not getattr(result, "chain", None):
            return
        self._remember_client(event)
        self._mark_woken(event)
        self_id = str(event.get_self_id())
        keys = self._source_keys(event)
        if not keys:
            return

        if any(k in self.recalled for k in keys):
            try:
                event.clear_result()
            except Exception:
                pass
            event.stop_event()
            return

        kind, tid = self._target(event)
        p = {
            "kind": kind,
            "tid": tid,
            "self_id": self_id,
            "t_start": time.time(),
            "t_end": None,
            "ids": [],
            "revoked": False,
        }
        for k in keys:  # 合并消息：一批原消息共用同一份回复记录
            self.pending[k] = p

    # ---------- 4. 发送后：找出 Bot 刚发的消息 id ----------

    @filter.after_message_sent()
    async def after_send(self, event: AstrMessageEvent):
        if not self._on(event):
            return
        keys = self._source_keys(event)
        p = next((self.pending[k] for k in keys if k in self.pending), None)
        if not p or p["t_end"] is not None:
            return
        p["t_end"] = time.time()
        asyncio.create_task(self._claim_later(keys, p))

    async def _claim_later(self, keys, p):
        await asyncio.sleep(float(self._c("lookup_delay", 1.0)))
        found = await self._find_bot_msgs(p)
        p["ids"] = p["ids"] or found
        hit = next((k for k in keys if k in self.recalled), None)
        if hit:
            await self._revoke(hit, p)

    async def _find_bot_msgs(self, p):
        count = int(self._c("history_count", 20))
        if p["kind"] == GROUP:
            ret = await self._call(
                p["self_id"], "get_group_msg_history", group_id=int(p["tid"]), count=count
            )
        else:
            ret = await self._call(
                p["self_id"], "get_friend_msg_history", user_id=int(p["tid"]), count=count
            )
        msgs = ret.get("messages", []) if isinstance(ret, dict) else []
        lo = p["t_start"] - 2
        hi = (p["t_end"] or time.time()) + 2
        ids = []
        for m in msgs:
            sender = (m.get("sender") or {}).get("user_id", m.get("user_id", ""))
            if str(sender) != p["self_id"]:
                continue
            mid = str(m.get("message_id"))
            t = float(m.get("time", 0) or 0)
            if lo <= t <= hi and mid not in self.claimed and mid not in self.error_ids:
                ids.append(mid)
                self.claimed.add(mid)
        return ids

    # ---------- 5. 双撤 ----------

    async def _revoke(self, key, p):
        if p.get("revoked"):
            return
        p["revoked"] = True
        ids = p["ids"] or await self._find_bot_msgs(p)
        p["ids"] = ids
        ok = 0
        for i in ids:
            ret = await self._call(p["self_id"], "delete_msg", message_id=int(i))
            if ret is not None:
                ok += 1
        if self._c("debug", False):
            logger.info(f"[RecallSync] revoked {ok}/{len(ids)} bot message(s) for {key}")

    # ---------- 6. 伪报错（可折叠） ----------

    def _fold_error(self, self_id, kind, tid):
        if not self._c("fake_error", True) or not tid:
            return
        base = f"{self_id}:{kind}:{tid}"
        if float(self._c("fold_window", 6) or 0) <= 0:
            fk = f"{base}:{random.getrandbits(32)}"
            self.fold[fk] = {"n": 1}
            asyncio.create_task(self._flush(str(self_id), kind, str(tid), fk))
            return
        f = self.fold.get(base)
        if f:
            f["n"] += 1
            return
        self.fold[base] = {"n": 1}
        asyncio.create_task(self._flush(str(self_id), kind, str(tid), base))

    async def _flush(self, self_id, kind, tid, fk):
        window = float(self._c("fold_window", 6) or 0)
        await asyncio.sleep(window if window > 0 else 1.0)
        f = self.fold.pop(fk, None)
        if not f:
            return
        tpls = self._c("error_templates", DEFAULT_TEMPLATES) or DEFAULT_TEMPLATES
        trace = "%08x" % random.getrandbits(32)
        try:
            text = random.choice(tpls).format(
                n=f["n"], trace=trace, time=time.strftime("%H:%M:%S")
            )
        except Exception:
            text = DEFAULT_TEMPLATES[0].format(n=f["n"], trace=trace)

        if str(self._c("error_style", "forward")) == "forward":
            # 和 AstrBot 超长消息折叠成合并转发的格式一模一样
            node = {
                "type": "node",
                "data": {
                    "user_id": str(self_id),
                    "nickname": str(self._c("forward_nickname", "AstrBot")),
                    "content": [{"type": "text", "data": {"text": text}}],
                },
            }
            if kind == GROUP:
                ret = await self._call(
                    self_id, "send_group_forward_msg", group_id=int(tid), messages=[node]
                )
            else:
                ret = await self._call(
                    self_id, "send_private_forward_msg", user_id=int(tid), messages=[node]
                )
        elif kind == GROUP:
            ret = await self._call(
                self_id, "send_group_msg", group_id=int(tid), message=text, auto_escape=True
            )
        else:
            ret = await self._call(
                self_id, "send_private_msg", user_id=int(tid), message=text, auto_escape=True
            )
        if isinstance(ret, dict) and ret.get("message_id") is not None:
            eid = str(ret["message_id"])
            self.error_ids.add(eid)
            ttl = float(self._c("error_ttl", 0) or 0)
            if ttl > 0:
                await asyncio.sleep(ttl)
                await self._call(self_id, "delete_msg", message_id=int(eid))
