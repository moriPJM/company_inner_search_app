"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# Streamlit Cloud対応: secretsからOpenAI APIキーを取得
# ローカル環境では.envファイルの値を優先し、Streamlit Cloudでのみsecretsを使用
try:
    # ローカル環境で有効なAPIキーが既に設定されている場合は、それを優先
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY').startswith('your-'):
        # Streamlit Cloudでのみ有効
        if 'streamlit' in globals() and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            if not st.secrets['OPENAI_API_KEY'].startswith('your-'):
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
except:
    # ローカル環境では.envファイルが使用される
    pass


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if hasattr(st.session_state, "retriever") and st.session_state.retriever is not None:
        return
    
    try:
        # RAGの参照先となるデータソースの読み込み
        docs_all = load_data_sources()
        
        # 同一ファイルから複数ドキュメントが生成された場合の統合処理
        docs_all = consolidate_documents_by_source(docs_all)
        
        # 重要なドキュメント（社員名簿など）を優先して含める
        docs_all = prioritize_important_documents(docs_all)

        # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
        for doc in docs_all:
            doc.page_content = adjust_string(doc.page_content)
            for key in doc.metadata:
                doc.metadata[key] = adjust_string(doc.metadata[key])
        
        # エンベディングモデルの初期化（フォールバック対応）
        logger.info("Initializing embeddings with fallback strategy")
        embeddings = None
        
        # 環境変数でHuggingFaceをスキップするオプション
        skip_huggingface = os.getenv('SKIP_HUGGINGFACE', 'false').lower() == 'true'
        
        if not skip_huggingface:
            # ローカル埋め込みモデルの使用を試行
            logger.info("Attempting to use lightweight local embeddings")
            try:
                import warnings
                
                # PyTorchの問題を回避するため、環境変数を設定
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Local embeddings model loaded successfully")
                
            except Exception as e:
                logger.error(f"Local embeddings failed: {e}")
                embeddings = None
        
        # OpenAIエンベディングにフォールバック
        if embeddings is None:
            logger.info("Falling back to OpenAI embeddings")
            try:
                embeddings = OpenAIEmbeddings()
                logger.info("OpenAI embeddings loaded successfully")
            except Exception as api_error:
                logger.error(f"OpenAI embeddings also failed: {api_error}")
                embeddings = None
        
        # キーワードベース検索にフォールバック
        if embeddings is None:
            logger.info("Falling back to keyword-based search")
            retriever = create_simple_keyword_retriever(docs_all)
            st.session_state.retriever = retriever
            logger.info("Keyword-based retriever initialized successfully")
            return

        # チャンク分割用のオブジェクトを作成
        text_splitter = CharacterTextSplitter(
            chunk_size=ct.CHUNK_SIZE,
            chunk_overlap=ct.CHUNK_OVERLAP,
            separator=ct.CHUNK_SEPARATOR
        )

        # 重要なドキュメント（社員名簿など）は分割せず、その他のドキュメントのみ分割
        splitted_docs = []
        important_keywords = ['社員名簿.csv', '議事録ルール.txt']
        
        for doc in docs_all:
            source = doc.metadata.get('source', '')
            is_important = any(keyword in source for keyword in important_keywords)
            
            if is_important:
                # 重要なドキュメントは分割せずにそのまま追加
                splitted_docs.append(doc)
                logger.info(f"Keeping important document unsplit: {source} ({len(doc.page_content)} chars)")
            else:
                # その他のドキュメントは通常通り分割
                chunks = text_splitter.split_documents([doc])
                splitted_docs.extend(chunks)
                logger.info(f"Split document: {source} into {len(chunks)} chunks")
        
        logger.info(f"Total documents after processing: {len(splitted_docs)}")
        
        # Chromaデータベース用にメタデータを整理（リストや複雑なオブジェクトを文字列に変換）
        for doc in splitted_docs:
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    doc.metadata[key] = ", ".join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool)):
                    doc.metadata[key] = str(value)

        # ベクターストアの作成
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)

        # ベクターストアを検索するRetrieverの作成（より多くの結果を取得して精度向上）
        # 社員名簿のような重要文書を確実に取得するため、k値を増やす
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": 10})
        
        # 社員名簿専用の高精度検索のため、ベクトルストアも保存
        st.session_state.vectorstore = db
        
        logger.info("Retriever initialized successfully")
        
    except Exception as e:
        logger.error(f"Critical error in retriever initialization: {str(e)}")
        
        # 最終フォールバック: キーワードベース検索
        try:
            logger.info("Attempting final fallback to keyword-based search")
            docs_all = load_data_sources()
            docs_all = consolidate_documents_by_source(docs_all)
            docs_all = prioritize_important_documents(docs_all)
            
            retriever = create_simple_keyword_retriever(docs_all)
            st.session_state.retriever = retriever
            logger.info("Final fallback successful - keyword-based retriever initialized")
            
        except Exception as fallback_error:
            logger.error(f"Final fallback also failed: {fallback_error}")
            # 空のretrieverを設定して完全な失敗を防ぐ
            st.session_state.retriever = None
            raise Exception(f"Complete initialization failure: {str(e)}, Fallback error: {str(fallback_error)}")
def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def consolidate_documents_by_source(docs_all):
    """
    同一ファイルから読み込まれた複数のドキュメントを1つに統合する
    
    Args:
        docs_all: 読み込んだドキュメントのリスト
        
    Returns:
        統合されたドキュメントのリスト
    """
    from collections import defaultdict
    
    # ファイルソース別にドキュメントをグループ化
    docs_by_source = defaultdict(list)
    for doc in docs_all:
        source = doc.metadata.get("source", "unknown")
        docs_by_source[source].append(doc)
    
    consolidated_docs = []
    for source, source_docs in docs_by_source.items():
        if len(source_docs) == 1:
            # 1つのドキュメントのみの場合はそのまま追加
            consolidated_docs.extend(source_docs)
        else:
            # 複数のドキュメントがある場合は統合
            combined_content = []
            combined_metadata = source_docs[0].metadata.copy()
            
            for i, doc in enumerate(source_docs):
                combined_content.append(f"=== セクション {i+1} ===")
                combined_content.append(doc.page_content)
                combined_content.append("")  # 空行で区切り
            
            # 統合ドキュメントを作成
            from langchain.schema import Document
            consolidated_doc = Document(
                page_content="\n".join(combined_content),
                metadata=combined_metadata
            )
            consolidated_docs.append(consolidated_doc)
    
    return consolidated_docs


def prioritize_important_documents(docs_all):
    """
    重要なドキュメント（社員名簿、議事録ルールなど）を優先して配置する
    
    Args:
        docs_all: ドキュメントのリスト
        
    Returns:
        優先度順に並び替えられたドキュメントのリスト
    """
    # 重要なファイルのキーワード（優先度順）
    important_keywords = [
        "社員名簿.csv",
        "議事録ルール.txt",
        "サービスについて",
        "会社について"
    ]
    
    priority_docs = []
    other_docs = []
    
    # 重要なドキュメントを優先リストに追加
    for keyword in important_keywords:
        for doc in docs_all:
            source = doc.metadata.get("source", "")
            if keyword in source and doc not in priority_docs:
                priority_docs.append(doc)
                break  # 各キーワードにつき1つのドキュメントのみ
    
    # 残りのドキュメントを追加
    for doc in docs_all:
        if doc not in priority_docs:
            other_docs.append(doc)
    
    # 優先ドキュメントを先頭に配置
    return priority_docs + other_docs


def create_simple_keyword_retriever(docs_all):
    """
    埋め込みベクトルを使わない簡易キーワードベース検索システムを作成
    
    Args:
        docs_all: ドキュメントのリスト
    """
    import streamlit as st
    import logging
    
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    class SimpleKeywordRetriever:
        def __init__(self, documents):
            self.documents = documents
            
        def invoke(self, query, k=5):
            """
            キーワードベースの簡易検索
            """
            query_lower = query.lower()
            scored_docs = []
            
            for doc in self.documents:
                content_lower = doc.page_content.lower()
                source_lower = doc.metadata.get("source", "").lower()
                
                # 基本スコア計算
                score = 0
                query_words = query_lower.split()
                
                for word in query_words:
                    # コンテンツ内の出現回数
                    content_count = content_lower.count(word)
                    score += content_count * 1
                    
                    # ソース名での一致（高い重み）
                    if word in source_lower:
                        score += 10
                
                # 特別なキーワード処理
                if "人事" in query_lower and "社員名簿" in source_lower:
                    score += 50
                if "議事録" in query_lower and "議事録ルール" in source_lower:
                    score += 50
                    
                if score > 0:
                    scored_docs.append((score, doc))
            
            # スコア順にソートして上位k件を返す
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:k]]
    
    # 簡易検索システムをセッション状態に保存
    retriever = SimpleKeywordRetriever(docs_all)
    logger.info("Simple keyword-based retriever created successfully")
    return retriever


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader_func = ct.SUPPORTED_EXTENSIONS[file_extension]
        result = loader_func(path)
        
        # 戻り値がリスト（カスタムローダーの場合）か、ローダーオブジェクトかを判定
        if isinstance(result, list):
            # カスタムローダーの場合、直接ドキュメントリストが返される
            docs = result
        else:
            # 通常のローダークラスの場合
            docs = result.load()
        
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s