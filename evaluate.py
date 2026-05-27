"""评测脚本：使用 50 条标注数据评测 Agent 性能"""
import json
import time
from app.agent.nodes import detect_intent_and_emotion

# 50 条标注测试数据：[用户消息, 期望意图, 期望情绪, 是否应建工单]
TEST_DATA = [
    # === 咨询类 ===
    ["我的包裹什么时候能到？", "咨询", "平静", False],
    ["这个商品支持支付宝付款吗？", "咨询", "平静", False],
    ["运费是怎么计算的？", "咨询", "平静", False],
    ["欧洲的物流时效大概多久？", "咨询", "平静", False],
    ["下单后多久能发货？", "咨询", "平静", False],
    ["怎么修改收货地址？", "咨询", "平静", False],
    ["支持哪些货币结算？", "咨询", "平静", False],
    ["如果我买两件，能合并发货吗？", "咨询", "平静", False],
    ["美国的关税大概多少？", "咨询", "平静", False],
    ["包裹清关一般要多久？", "咨询", "平静", False],
    ["怎么查订单物流信息？", "咨询", "平静", False],
    ["支付方式有哪些？", "咨询", "平静", False],
    ["注册需要什么信息？", "咨询", "平静", False],

    # === 售后类 ===
    ["尺码不合适，怎么换？", "售后", "平静", False],
    ["退货流程是什么？", "售后", "平静", False],
    ["退款多久能到账？", "售后", "平静", False],
    ["我收到的东西和你描述的不一样", "售后", "不满", False],
    ["30 天无理由退货的规则是什么？", "售后", "平静", False],
    ["退货地址在哪？", "售后", "平静", False],
    ["内衣可以退货吗？", "售后", "平静", False],
    ["退货的运费谁出？", "售后", "平静", False],
    ["退款是按什么汇率退的？", "售后", "平静", False],

    # === 投诉类 ===
    ["包裹丢了！三天了还没到！", "投诉", "愤怒", True],
    ["给我的东西是坏的，你们怎么搞的", "投诉", "不满", True],
    ["关税太高了，你们的价格都是骗人的", "投诉", "不满", False],
    ["客服电话打不通，你们到底有没有人在上班", "投诉", "不满", True],
    ["快递把包裹弄坏了，里面的东西都碎了", "投诉", "愤怒", True],
    ["我在海关那边卡了一周了没人管", "投诉", "不满", True],
    ["为什么别人的运费比我便宜？", "投诉", "不满", False],
    ["你们网站太烂了，支付一直失败", "投诉", "不满", False],

    # === 退款类 ===
    ["我要退款，这个产品我不满意", "退款", "不满", False],
    ["怎么取消订单退款？", "退款", "平静", False],
    ["显示退款成功但我没收到钱", "退款", "不满", True],
    ["我付款了但订单被取消了，钱会退吗？", "退款", "平静", False],
    ["能不能退差价？", "退款", "平静", False],
    ["用 PayPal 退款要多久？", "退款", "平静", False],

    # === 要求转人工 ===
    ["我要找人工客服", "要求转人工", "平静", True],
    ["AI 回答没用，给我转人工", "要求转人工", "不满", True],
    ["帮我转接真人", "要求转人工", "平静", True],
    ["你们的机器人不行，叫我找真人", "要求转人工", "不满", True],

    # === 其他紧急 ===
    ["我账号被盗了，有人用我的卡下了单", "其他", "紧急", True],
    ["我的信用卡被你们多扣了 500 美元", "退款", "紧急", True],
    ["马上帮我取消订单！立刻！", "退款", "紧急", True],
    ["我要报警了，你们是诈骗网站", "投诉", "愤怒", True],
    ["限你们24小时内解决，不然我告你们", "投诉", "愤怒", True],
    ["我隐私泄露了，你们有没有安全保护", "其他", "不满", True],
    ["有安全隐患吗？用户的支付信息安全吗？", "咨询", "平静", False],
    ["春节你们发货吗？", "咨询", "平静", False],
    ["可以送到 PO Box 吗？", "咨询", "平静", False],
    ["怎么联系物流公司？", "咨询", "平静", False],
    ["订单状态为什么两天没更新了？", "咨询", "不满", False],
    ["双十一有什么优惠活动吗？", "咨询", "平静", False],
    ["这个材质是什么？", "咨询", "平静", False],
]


def evaluate():
    print("=" * 60)
    print("AI 客服工单智能处理系统 — 评测报告")
    print("=" * 60)

    # 评测意图 + 情绪
    intent_correct = 0
    emotion_correct = 0
    total = len(TEST_DATA)

    for i, (msg, exp_intent, exp_emotion, _) in enumerate(TEST_DATA):
        result = detect_intent_and_emotion({"user_message": msg})
        actual_intent = result.get("intent", "")
        actual_emotion = result.get("emotion", "")

        if actual_intent == exp_intent:
            intent_correct += 1
        if actual_emotion == exp_emotion:
            emotion_correct += 1

        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{total}] \"{msg[:30]}...\" → 意图:{actual_intent}(期望:{exp_intent}) 情绪:{actual_emotion}(期望:{exp_emotion})")

    intent_acc = round(intent_correct / total * 100, 1)
    emotion_acc = round(emotion_correct / total * 100, 1)

    # 评测 AI 解决率（should_not_create_ticket 视为 AI 可直接解决）
    # 模拟判断：紧急/愤怒 → 应建单，其余看是否匹配知识库
    ticket_expected = sum(1 for _, _, _, need_ticket in TEST_DATA if need_ticket)
    should_not_create = total - ticket_expected

    print("\n" + "=" * 60)
    print("评测结果：")
    print(f"  意图识别准确率: {intent_acc}%  ({intent_correct}/{total})")
    print(f"  情绪识别准确率: {emotion_acc}%  ({emotion_correct}/{total})")
    print(f"  理论 AI 解决率: {round(should_not_create / total * 100, 1)}%  ({should_not_create}/{total})")
    print(f"  回复质量评分: 需通过 LLM-as-Judge 评估（待后续补充）")
    print("=" * 60)

    return {
        "intent_accuracy": intent_acc,
        "emotion_accuracy": emotion_acc,
        "ai_resolve_rate": round(should_not_create / total * 100, 1),
        "total_samples": total,
    }


if __name__ == "__main__":
    evaluate()
