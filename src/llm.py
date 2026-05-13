import json
import os
import numpy as np
from openai import OpenAI


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

SYSTEM_PROMPT = """你是一个港股技术分析助手,帮助一位看不懂K线的用户理解股票数据。

约定:
- 输出必须是简短易懂的中文,避免专业术语堆砌,必要时用一句话解释术语。
- 不要给出"买入/卖出"指令,只描述当前位置和技术参考区间,决策权在用户。
- 历史高低点要简要说明可能的市场背景(如有公开信息),不强行编造原因。
- 输出格式严格遵守用户指定的字段结构。"""


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )


def _call(prompt: str, model: str) -> str:
    resp = _client().chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def morning_brief(name: str, symbol: str, analysis: dict, model: str, effort: str = "medium") -> str:
    prompt = f"""为关注股票生成今日盘前简报。读者**完全看不懂K线**,你必须用大白话解释每个技术信号背后的意思。

股票: {name} ({symbol})

技术数据(JSON):
{_dumps(analysis)}

严格按以下五节输出, 每节标题用 ** 包裹, 不要加多余引言或总结:

**昨日K线**
2-3句话翻译 kline 字段:涨跌幅、K线形态的含义(用一句话解释这个形态意味着什么)、量能变化(放量还是缩量, 量比说明什么)。

**近两年位置**
读 range 字段。说明 current 在 abs_low~abs_high 区间的位置(percentile 是分位数, 50%中间, 100%顶), drawdown_from_high 从最高点回撤多少, rebound_from_low 从最低点反弹多少。然后**分别**提到 top_highs 的两个高点和 top_lows 的两个低点(日期+价格), 如果是知名时间节点(如2024年初港股反弹、2025年10月美股新高传导等)简要说明可能背景, 不知道就跳过不要硬编。

**技术指标解读**
读 indicators 字段, 翻译给小白看, 每个指标 1-2 句:
- MA(均线): 看 arrangement 是多头/空头/纠缠, 解释当前价格 vs MA20/MA60 高低意味着什么趋势。
- BOLL(布林带): 看 position 字段, 告诉用户当前在上轨/中轨/下轨什么位置, 含义是什么(超买/超卖/偏强/偏弱)。
- MACD: 看 signal 字段, 翻译金叉/死叉/零轴上下是什么意思(多空力量、动能变化)。
- KDJ: 看 signal 字段, K/D/J 值简单说一下短线状态(超买区、超卖区、金叉、死叉)。

**关键技术位**
support_resistance 里 nearest_support / nearest_resistance 各报一个具体价位 + 距离当前价百分比。再提一句 key_supports / key_resistances 的其他位置。

**参考买入区间**
直接给出 value_zone 的 zone_low~zone_high 数字, 把 anchor_desc 和 method 翻译给小白看(锚定到哪个支撑位、为什么这么算)。读 position_desc 字段告诉用户当前位置含义:
- in_zone 为 true → "当前已进入参考买入区间"
- position 为 above_zone → "需等回调"
- position 为 below_zone → "已破位, 需谨慎"

末尾加一句: "这是基于历史数据的技术参考价位, 不是投资建议, 决策权在你。"
"""
    return _call(prompt, model)


def closing_review(name: str, symbol: str, analysis: dict, model: str, effort: str = "medium") -> str:
    prompt = f"""为关注股票生成今日盘后简报, 读者看不懂K线, 用大白话。

股票: {name} ({symbol})

技术数据(JSON):
{_dumps(analysis)}

按以下四节输出, 每节标题用 ** 包裹:

**今日表现**
2-3句话翻译 kline:涨跌幅、形态含义、量能(放量/缩量)。

**位置变化**
今日收盘在近两年区间的什么位置(range.percentile), 离 support_resistance 里的最近支撑/阻力有多远。

**指标信号**
1-2句话总结 indicators:重点说 MACD 的 signal、KDJ 的 signal、BOLL 的 position 三项里**最值得关注**的(比如金叉/死叉/超买/超卖发生了说一下,没异动就说"指标维持现状")。

**明日关注点**
如果今日接近关键支撑/阻力位, 说一句要观察什么(突破/跌破后的潜在方向)。否则简短带过即可。"""
    return _call(prompt, model)


def snapshot_summary(rows: list, model: str) -> str:
    """读全部股票快照, 输出 2-3 句"今日异动"总结. 没异动就一句话带过."""
    prompt = f"""下面是今日盘后关注列表 {len(rows)} 只股票的技术数据快照(JSON):

{_dumps(rows)}

请用 **2-3 句中文** 只点出今天**最值得关注的 1-3 件事**, 不要逐只复述:
- 谁触发了关键信号(MACD 金叉/死叉, KDJ 金叉/死叉, BOLL 突破上下轨)
- 谁进入或离开了参考买入区间
- 谁逼近重要支撑/阻力位
- 谁涨跌幅或量能特别异常(>3% 或量比>2 算异常)

如果今天全部平淡无异动, 就一句"今日各股指标维持现状, 无明显信号变化"即可。
直接给结果, 不要标题、不要前言、不要列表项。"""
    return _call(prompt, model)


def intraday_event_alert(name: str, symbol: str, price: float, events: list,
                         analysis: dict, model: str) -> str:
    """events 是 [(event_id, 中文描述), ...] 列表"""
    event_list = "\n".join(f"- {desc}" for _, desc in events)
    prompt = f"""股票 {name} ({symbol}) 现价 {price}, 盘中刚刚触发以下技术事件:

{event_list}

简要技术数据:
{_dumps(analysis)}

请用 **3-4 句中文** 给小白看的口语化解读:
1) 把上述事件用大白话说清楚(为什么会触发, 数字含义)
2) 这意味着短线什么状态(强势/弱势/方向不明)
3) 关键位置(下个支撑/阻力在哪, 距离多少)
4) 一句温馨提示: "这是技术触发提示, 不是买卖建议"

直接给结果, 不要标题不要前言。"""
    return _call(prompt, model)
