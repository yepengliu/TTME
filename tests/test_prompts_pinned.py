"""Pinned memory, memo and card prompts: changing one, or adding one, must update PINNED."""
import hashlib
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "ttt" / "prompts"
DIRS = ("cards", "memo", "memory")
PINNED = {
    "cards/deepen_v1.txt": "d9987a17da0cfd17710e752dba05bc0ad758390927d8d6b9d581c2e9207b8fe9",
    "cards/deepen_v2.txt": "a17b9c5ed549b95b37b3f2b2c88a70f0242de398adaab0cdc949e280a178cb30",
    "cards/none.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "memo/assigned_direction.txt": "054bab3d4e0bc6046cb5cc0173caf21a22e0c02da861389588cf08739690949c",
    "memo/family_memory_shape.txt": "48bf3eea5878083ad84824904e9c3cd8c0634f80f0694ef5728a231212b1aae9",
    "memo/family_rewrite.txt": "fffc78119c3bb4b19d478455d5411ab337b4a0e3a3045e023932f8e1484d3684",
    "memo/family_start.txt": "f905edfbbc5dd5e9bb9146e7900378c9ca8f02962f6079d9df7c645cefb6c1b9",
    "memo/lineage_split.txt": "e6f262e71827f14beafc6bf8b6dc6391fe71970f8c527383333e201eadb58ffa",
    "memo/lineage_steered.txt": "cc1708c57eada2f32b07be690ca260878630f1bc84a5aadf6bf05a26af4420f8",
    "memo/lineage_unsteered.txt": "916d8a051a513330d3814a713a9c86f6380845bb9d01b50690e38c005c730e4c",
    "memo/merge.txt": "b734d8c7cb362399a1156c3c984bbd7132bb7787986df28ad57475ea73eb9217",
    "memo/merge_retry_prefix.txt": "079697473b3dfe73b0eb830a131a4833c4de31feb46b9ec5477b2043fb724dcd",
    "memo/mine.txt": "a81ecdde5031785d46611f429bb3626f825677cdd2ed6aabcd3eed64b999cbf1",
    "memo/mine_pair.txt": "80579f912f751a0cec3233f845f2e468ffe5c38908303721b564a89954deaf37",
    "memo/mine_pair_refused.txt": "d26715c9dfaad161f8e6fb2d80f7ca3d22c637b93d27064b94dc3a02ee0727da",
    "memo/score_history.txt": "8267a27d606b30e8541fd3533e32c43be2ddf9f9d871661603cf31231e746368",
    "memo/select.txt": "5b99bb9158cdc977da7109f4bda2eb75445dea1f04e5d07f91233e72613c5569",
    "memo/select_k.txt": "89b7279d807def12077dd096a7ebbb0e0f5f6986b6b3119740498be15c15640d",
    "memo/select_k_serial.txt": "4e807513cd1b02182ebb8e9d71159e7078d49bf0946bce46cdb3ba941be6c5dc",
    "memo/select_serial.txt": "d4536eb4f7000d777114b655702ba9ed5890b7a51631eb50a278fff847f8555c",
    "memo/write.txt": "da9671e8b8f327286b8b5ef67375b6c88698962e79768a65bdc568fff284eb37",
    "memory/crash_attribution_guide.txt": "4b1885724185129ff57d15a85acde88f86d8bb106022cfe08b106998cf78e89c",
    "memory/crash_code_guide.txt": "cd520f96640b6bf7f4dfb65199830f343ded15193864fcac2fc0ee845717b957",
    "memory/distill_question.txt": "bb85ec0b18f8a4eea2a62eac6c7fab544d9240d37c92ee8fa1f7fc96dfaac354",
    "memory/guideline_heading.txt": "4a1f22571e953c56286c79934c83bdb99fceb71b88885b5e89556d7312822605",
    "memory/guideline_update.txt": "0e8e7f499ca3cd42a0ce1239e8673e13d0787a1a0867caf1dc038452c3b0c2fd",
    "memory/microdistill_synthesis_guide.txt": "1557ec751ad7114fddbf535a393302ec4303b8e4b0e092014a1c3b5f5584c092",
    "memory/microdistill_system.txt": "c01c60c3e62e8b7ed9b0d2320de8a7f2a43aa7660bb14ee76b5051408d736946",
    "memory/recipe_heading.txt": "645ebe0b7fedb53f8c204ecdc739e2dc2565d11dd47beb65e435a8a759fe292c",
}


def _current() -> dict[str, str]:
    return {p.relative_to(PROMPTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in DIRS for p in sorted((PROMPTS / d).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def test_prompt_files_match_their_pins():
    current = _current()
    changed = {name: digest for name, digest in current.items() if PINNED.get(name) != digest}
    missing = sorted(set(PINNED) - set(current))
    assert not changed and not missing, (
        f"prompt files changed or added (new sha256 values): {changed}; removed: {missing}")


def test_guideline_update_prompt_opening():
    text = (PROMPTS / "memory" / "guideline_update.txt").read_text(encoding="utf-8")
    assert text.startswith("You maintain a compact Guideline Memory")
