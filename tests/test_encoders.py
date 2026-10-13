"""Real tiny Transformers backbones: no remote weights and no mocked forward pass."""
import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (EsmConfig, EsmModel, EsmTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizerFast)

from src.protein_encoder import ProteinEncoder
from src.ligand_encoder import LigandEncoder


@pytest.fixture
def small_backbones(tmp_path):
    torch.set_num_threads(2)
    protein, ligand = tmp_path / "esm", tmp_path / "chem"
    protein.mkdir()
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", "A", "C", "G", "<mask>"]
    vocab_path = protein / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    EsmTokenizer(vocab_file=str(vocab_path)).save_pretrained(protein)
    EsmModel(EsmConfig(vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
                       num_attention_heads=3, intermediate_size=24, max_position_embeddings=34,
                       pad_token_id=1, mask_token_id=7), add_pooling_layer=False).save_pretrained(protein)
    ligand.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["CCO", "CCN", "C=O"], vocab_size=270,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer_path = ligand / "tokenizer.json"
    bpe.save(str(tokenizer_path))
    RobertaTokenizerFast(tokenizer_file=str(tokenizer_path), bos_token="<s>", eos_token="</s>",
                         sep_token="</s>", cls_token="<s>", unk_token="<unk>",
                         pad_token="<pad>", mask_token="<mask>").save_pretrained(ligand)
    RobertaModel(RobertaConfig(vocab_size=bpe.get_vocab_size(), hidden_size=12, num_hidden_layers=1,
                               num_attention_heads=3, intermediate_size=24, max_position_embeddings=34),
                 add_pooling_layer=False).save_pretrained(ligand)
    return str(protein), str(ligand)


def test_frozen_encoder_cache_and_chunking(small_backbones, tmp_path):
    protein_path, ligand_path = small_backbones
    p = ProteinEncoder(protein_path, cache=tmp_path / "p_cache", chunk_size=4)
    l = LigandEncoder(ligand_path, cache=tmp_path / "l_cache", max_length=16)
    p.train()
    l.train()
    assert not p.backbone.training and not l.backbone.training
    residues = p("ACGAAC")
    ligand = l("CCO")
    assert residues.shape == (6, 12) and ligand.shape == (12,)
    assert not residues.requires_grad and not ligand.requires_grad
    assert torch.equal(residues, p("ACGAAC"))
    assert torch.equal(ligand, l("CCO"))
    assert torch.equal(residues.mean(0), p("ACGAAC", pooling="mean"))
    assert len(list((tmp_path / "p_cache").glob("*.pt"))) == 1
    with pytest.raises(ValueError, match="exceeds"):
        l("C" * 100)


def test_unfrozen_encoders_receive_gradients(small_backbones, tmp_path):
    protein_path, ligand_path = small_backbones
    for encoder, text in [(ProteinEncoder(protein_path, freeze=False, cache=tmp_path, chunk_size=4), "ACGAAC"),
                           (LigandEncoder(ligand_path, freeze=False, cache=tmp_path, max_length=16), "CCO")]:
        assert encoder.cache is None
        result = encoder(text)
        assert result.requires_grad
        # Nonconstant loss: LayerNorm output sum alone would have zero gradients.
        (result * torch.randn_like(result)).sum().backward()
        grads = [p.grad for p in encoder.backbone.parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum().item() for g in grads) > 0
