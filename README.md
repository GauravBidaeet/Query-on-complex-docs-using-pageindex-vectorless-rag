#Complex Document QA System (PageIndex RAG)

This is a vectorless RAG system that retrieves precise answers from legal documents using page-level indexing — no embeddings required.

 How It Works?
 LexIndex Pipeline : Legal Documents  ──►  Page Indexer  ──►  Page Index Store(PDF/TXT)                                            
                     User Query  ──►  Index Matcher  ──►  Relevant Pages  ──►   LLM (GPT/Claude)  ──►   Grounded Legal Answer

1.Indexing Phase — Legal documents are parsed and split into pages. Each page is indexed with metadata (document name, page number, section title if available).
2.Retrieval Phase — A user query is matched against the page index using keyword and structural matching to find the most relevant pages.
3.Generation Phase — The retrieved pages are passed as context to an LLM, which generates a precise, cited answer.
