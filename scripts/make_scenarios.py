"""Generate the assessment's test scenarios as synthetic videos with ground truth.

    python scripts/make_scenarios.py            # all scenarios -> samples/scenarios/
    python scripts/make_scenarios.py noisy      # just one

The two-speaker-with-music scenario is samples/two_speakers_music.mp4
(scripts/make_test_video.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_video import FEMALE, MALE, MALE2, build  # noqa: E402

OUT = Path("samples/scenarios")

SCENARIOS: dict[str, dict] = {
    # One speaker, clear Mandarin, no music.
    "one_speaker_clear": dict(music=False, dialogue=[
        (FEMALE, "大家好，欢迎来到我们的节目。", 0.8),
        (FEMALE, "今天我想和大家聊一聊，怎样在忙碌的生活中照顾好自己。", 0.8),
        (FEMALE, "首先，每天一定要保证充足的睡眠，至少七个小时。", 0.7),
        (FEMALE, "其次，饮食要均衡，多吃蔬菜和水果，少吃油炸食品。", 0.7),
        (FEMALE, "最后，别忘了每周运动三次，哪怕只是散散步也好。", 0.8),
        (FEMALE, "希望这些小建议对你有帮助，我们下期再见。", 1.0),
    ]),
    # Fast Mandarin: edge-tts rate +35%.
    "fast_dialogue": dict(rate="+35%", dialogue=[
        (MALE, "快点快点，车马上就要开了，你东西都收拾好了没有？", 0.3),
        (FEMALE, "马上就好，我在找我的充电器，刚才明明放在桌子上的。", 0.3),
        (MALE, "别找了，路上再买一个，再不走我们就赶不上了！", 0.3),
        (FEMALE, "好好好，我拿上包就走，你先去楼下叫车吧。", 0.3),
        (MALE, "车已经叫好了，司机说三分钟以后就到门口。", 0.3),
        (FEMALE, "那你帮我拿一下这个箱子，太重了我一个人搬不动。", 0.3),
        (MALE, "给我吧，你把门锁好，钥匙别忘了带。", 0.3),
        (FEMALE, "放心吧，钥匙在我口袋里，走吧走吧。", 1.0),
    ]),
    # Long silences between lines: no speech must be generated there.
    "silent_sections": dict(dialogue=[
        (MALE, "这里就是我小时候住过的地方。", 7.0),
        (FEMALE, "变化真大，我几乎认不出来了。", 6.5),
        (MALE, "那棵老树还在，我们以前经常在下面乘凉。", 7.5),
        (FEMALE, "以后我们常回来看看吧。", 3.0),
    ]),
    # Long sentences: pacing without drift.
    "long_sentences": dict(dialogue=[
        (MALE2, "在过去的十年里，我们公司从一个只有五个人的小团队，发展成为拥有两千多名员工、"
                "业务覆盖全国三十多个城市的大型企业，这离不开每一位同事的辛勤付出。", 1.0),
        (FEMALE, "我完全同意，但是我们也必须清醒地认识到，市场环境正在发生深刻的变化，"
                 "如果我们不能及时调整战略、加快技术创新，就很可能在未来的竞争中被淘汰。", 1.0),
        (MALE2, "所以我建议，从明年开始，我们把研发投入提高到营业收入的百分之十五，"
                "同时在上海和深圳分别设立新的研发中心，吸引更多优秀的人才加入我们。", 1.5),
    ]),
    # Noisy audio: white noise at -20 dB relative to speech, plus music.
    "noisy": dict(noise_db=-20.0, dialogue=[
        (MALE, "喂，你能听见我说话吗？这边信号不太好。", 0.6),
        (FEMALE, "能听见，但是有点杂音，你现在在哪儿呢？", 0.6),
        (MALE, "我在火车站，外面人特别多，特别吵。", 0.6),
        (FEMALE, "那你几点到？我去车站接你。", 0.6),
        (MALE, "大概晚上八点半，到了我再给你打电话。", 0.6),
        (FEMALE, "好的，路上注意安全，到了记得给我发消息。", 1.5),
    ]),
}


def main(names: list[str]) -> None:
    for name in names or list(SCENARIOS):
        spec = dict(SCENARIOS[name])
        dialogue = spec.pop("dialogue")
        build(OUT / f"{name}.mp4", dialogue, **spec)


if __name__ == "__main__":
    main(sys.argv[1:])
