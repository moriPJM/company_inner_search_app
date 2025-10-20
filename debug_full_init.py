"""
完全な初期化処理のデバッグ用スクリプト
"""
import sys
import traceback
import os
import unicodedata
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

print("=== 完全初期化処理デバッグ ===")

try:
    # 必要なライブラリをインポート
    import constants as ct
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    print("1. データソースの読み込み...")
    
    def load_data_sources():
        docs_all = []
        
        def recursive_file_check(path, docs_all):
            if os.path.isdir(path):
                files = os.listdir(path)
                for file in files:
                    full_path = os.path.join(path, file)
                    recursive_file_check(full_path, docs_all)
            else:
                file_extension = os.path.splitext(path)[1]
                if file_extension in ct.SUPPORTED_EXTENSIONS:
                    try:
                        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
                        docs = loader.load()
                        docs_all.extend(docs)
                    except Exception as e:
                        print(f"  ⚠️ ファイル読み込みスキップ {os.path.basename(path)}: {e}")
        
        recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
        
        # Webページ読み込み
        web_docs_all = []
        for web_url in ct.WEB_URL_LOAD_TARGETS:
            try:
                loader = WebBaseLoader(web_url)
                web_docs = loader.load()
                web_docs_all.extend(web_docs)
            except Exception as e:
                print(f"  ⚠️ Webページ読み込みスキップ {web_url}: {e}")
        
        docs_all.extend(web_docs_all)
        return docs_all
    
    docs_all = load_data_sources()
    print(f"  ✓ 読み込み完了: {len(docs_all)} 件のドキュメント")
    
    print("2. Unicode正規化処理...")
    
    def adjust_string(s):
        if type(s) is not str:
            return s
        if sys.platform.startswith("win"):
            s = unicodedata.normalize('NFC', s)
            s = s.encode("cp932", "ignore").decode("cp932")
            return s
        return s
    
    doc_count = 0
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
        doc_count += 1
        
    print(f"  ✓ 正規化完了: {doc_count} 件のドキュメント")
    
    print("3. 埋め込みモデル初期化...")
    embeddings = OpenAIEmbeddings()
    print("  ✓ 埋め込みモデル OK")
    
    print("4. テキストスプリッター初期化...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    print("  ✓ テキストスプリッター OK")
    
    print("5. チャンク分割処理...")
    splitted_docs = text_splitter.split_documents(docs_all)
    print(f"  ✓ 分割完了: {len(splitted_docs)} 件のチャンク")
    
    print("6. ベクターストア作成...")
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    print("  ✓ ベクターストア作成 OK")
    
    print("7. Retriever作成...")
    retriever = db.as_retriever(search_kwargs={"k": 5})
    print("  ✓ Retriever作成 OK")
    
    print("8. 検索テスト...")
    test_results = retriever.get_relevant_documents("会社について")
    print(f"  ✓ 検索テスト OK: {len(test_results)} 件の結果")
    
    print("\n=== 完全初期化処理デバッグ完了 ===")
    print("すべての初期化処理が正常に完了しました。")

except Exception as e:
    print(f"\n✗ エラーが発生しました:")
    print(f"エラー: {e}")
    print(f"エラータイプ: {type(e).__name__}")
    print("\n詳細なトレースバック:")
    traceback.print_exc()