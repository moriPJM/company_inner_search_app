"""
初期化処理のデバッグ用スクリプト
"""
import sys
import traceback
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

print("=== 初期化処理デバッグ ===")

try:
    print("1. 基本ライブラリのインポート...")
    import os
    import logging
    from uuid import uuid4
    import unicodedata
    print("✓ 基本ライブラリ OK")
    
    print("2. Streamlitのインポート...")
    import streamlit as st
    print("✓ Streamlit OK")
    
    print("3. LangChain関連のインポート...")
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    print("✓ LangChain関連 OK")
    
    print("4. constants.pyのインポート...")
    import constants as ct
    print("✓ constants.py OK")
    
    print("5. OpenAI APIキーの確認...")
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key.startswith('sk-'):
        print("✓ OpenAI APIキー設定済み")
    else:
        print("✗ OpenAI APIキーが設定されていません")
    
    print("6. データディレクトリの確認...")
    if os.path.exists(ct.RAG_TOP_FOLDER_PATH):
        print(f"✓ データディレクトリ存在: {ct.RAG_TOP_FOLDER_PATH}")
    else:
        print(f"✗ データディレクトリが存在しません: {ct.RAG_TOP_FOLDER_PATH}")
    
    print("7. 埋め込みモデルのテスト...")
    embeddings = OpenAIEmbeddings()
    print("✓ 埋め込みモデル初期化 OK")
    
    print("8. テキストスプリッターのテスト...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    print("✓ テキストスプリッター初期化 OK")
    
    print("\n=== 初期化処理デバッグ完了 ===")
    print("すべてのコンポーネントが正常に動作しています。")

except Exception as e:
    print(f"\n✗ エラーが発生しました:")
    print(f"エラー: {e}")
    print(f"エラータイプ: {type(e).__name__}")
    print("\n詳細なトレースバック:")
    traceback.print_exc()