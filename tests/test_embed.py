from emoscript.embed import prefix_for


def test_prefix_for_model_families():
    assert prefix_for("intfloat/multilingual-e5-large") == "passage: "
    assert prefix_for("intfloat/e5-base-v2") == "passage: "
    assert prefix_for("nomic-ai/nomic-embed-text-v1.5") == "search_document: "
    assert prefix_for("BAAI/bge-base-en-v1.5") == ""
    assert prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""
    assert prefix_for("mixedbread-ai/mxbai-embed-large-v1") == ""
