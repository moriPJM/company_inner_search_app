"""
データ読み込み処理のデバッグ用スクリプト
"""
import sys
import traceback
import os
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

print("=== データ読み込みデバッグ ===")

try:
    # 必要なライブラリをインポート
    import constants as ct
    from langchain_community.document_loaders import WebBaseLoader
    
    print("1. データディレクトリの内容確認...")
    
    def check_directory_recursive(path, level=0):
        """ディレクトリの内容を再帰的に表示"""
        indent = "  " * level
        if os.path.isdir(path):
            print(f"{indent}📁 {os.path.basename(path)}/")
            try:
                items = os.listdir(path)
                for item in items:
                    full_path = os.path.join(path, item)
                    check_directory_recursive(full_path, level + 1)
            except PermissionError:
                print(f"{indent}  ❌ アクセス権限なし")
        else:
            file_extension = os.path.splitext(path)[1]
            if file_extension in ct.SUPPORTED_EXTENSIONS:
                print(f"{indent}📄 {os.path.basename(path)} (対応形式)")
            else:
                print(f"{indent}📄 {os.path.basename(path)} (未対応形式)")
    
    check_directory_recursive(ct.RAG_TOP_FOLDER_PATH)
    
    print("\n2. サポートされている拡張子...")
    for ext, loader in ct.SUPPORTED_EXTENSIONS.items():
        print(f"  {ext}: {loader}")
    
    print("\n3. Webページ読み込みテスト...")
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        print(f"  テスト中: {web_url}")
        try:
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            print(f"  ✓ 成功: {len(web_docs)} 件のドキュメント")
        except Exception as e:
            print(f"  ✗ 失敗: {e}")
    
    print("\n4. ファイル読み込みテスト...")
    docs_count = 0
    
    def test_file_load(path):
        global docs_count
        if os.path.isdir(path):
            try:
                files = os.listdir(path)
                for file in files:
                    full_path = os.path.join(path, file)
                    test_file_load(full_path)
            except Exception as e:
                print(f"  ✗ ディレクトリアクセスエラー {path}: {e}")
        else:
            file_extension = os.path.splitext(path)[1]
            if file_extension in ct.SUPPORTED_EXTENSIONS:
                try:
                    loader_class = ct.SUPPORTED_EXTENSIONS[file_extension]
                    loader = loader_class(path)
                    docs = loader.load()
                    docs_count += len(docs)
                    print(f"  ✓ {os.path.basename(path)}: {len(docs)} 件")
                except Exception as e:
                    print(f"  ✗ {os.path.basename(path)}: {e}")
    
    test_file_load(ct.RAG_TOP_FOLDER_PATH)
    print(f"\n総ドキュメント数: {docs_count}")
    
    print("\n=== データ読み込みデバッグ完了 ===")

except Exception as e:
    print(f"\n✗ エラーが発生しました:")
    print(f"エラー: {e}")
    print(f"エラータイプ: {type(e).__name__}")
    print("\n詳細なトレースバック:")
    traceback.print_exc()