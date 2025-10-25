"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# OSの操作を行うためのモジュール
import os
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME,
    layout="wide"
)

# カスタムCSSの適用
st.markdown("""
<style>
    /* サイドバーのスタイル調整 */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* メインコンテンツエリアの調整 */
    .main .block-container {
        padding-top: 2rem;
        max-width: none;
    }
    
    /* チャット入力欄のスタイル調整 */
    .stChatInput {
        margin-top: 1rem;
    }
    
    /* チャット入力欄の背景色を緑色に */
    .stChatInput > div {
        background-color: #e8f5e8 !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* 入力フィールドの背景色も緑色に */
    .stChatInput input {
        background-color: #f0f8f0 !important;
        border: 1px solid #90ee90 !important;
    }
    
    /* 全てのチャットメッセージコンテナを対象に */
    div[data-testid="stChatMessage"] {
        background-color: #d4edda !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* チャットメッセージ内の全てのテキストを緑色に */
    div[data-testid="stChatMessage"] * {
        color: #155724 !important;
    }
    
    /* 具体的なMarkdown要素も対象に */
    div[data-testid="stChatMessage"] .stMarkdown,
    div[data-testid="stChatMessage"] .stMarkdown *,
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] div {
        color: #155724 !important;
    }
    
    /* 警告メッセージも緑に統一 */
    div[data-testid="stChatMessage"] div[data-testid="stAlert"] {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
    }
    
    /* 警告メッセージのスタイル調整 */
    .stAlert[data-baseweb="notification"],
    div[data-testid="stAlert"] {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
    
    # 初期化成功の確認
    if not hasattr(st.session_state, 'retriever'):
        logger.warning("Retriever not found after initialization, creating fallback")
        st.session_state.retriever = None
        
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    
    # 詳細エラー情報を表示
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    
    # 初期化失敗時の詳細情報
    with st.expander("初期化エラーの詳細情報（管理者向け）"):
        st.code(f"""
エラー詳細: {str(e)}
環境変数SKIP_HUGGINGFACE: {os.getenv('SKIP_HUGGINGFACE', 'not set')}
環境変数TOKENIZERS_PARALLELISM: {os.getenv('TOKENIZERS_PARALLELISM', 'not set')}
""")
    
    # フォールバック初期化を試行
    try:
        logger.info("Attempting fallback initialization")
        # 最小限の初期化
        if "session_id" not in st.session_state:
            from uuid import uuid4
            st.session_state.session_id = uuid4().hex
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Retrieverをフォールバック状態に設定
        st.session_state.retriever = None
        
        st.warning("⚠️ 初期化が部分的に失敗しましたが、限定機能で動作を継続します。", icon="⚠️")
        
    except Exception as fallback_error:
        logger.error(f"Fallback initialization also failed: {fallback_error}")
        # 完全な失敗の場合のみ停止
        st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

# セッション状態の初期化
if "mode" not in st.session_state:
    st.session_state.mode = ct.ANSWER_MODE_1


############################################################
# 4. サイドバーの表示
############################################################
with st.sidebar:
    # 利用目的タイトル
    st.markdown("## 利用目的")
    
    # モード表示（サイドバーに移動）
    cn.display_select_mode()

############################################################
# 5. メインコンテンツエリアの表示
############################################################
# タイトル表示
cn.display_app_title()

# AIメッセージの初期表示
cn.display_initial_ai_message()

# 会話ログの表示
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

############################################################
# 6. チャット入力の受け付け（メインエリア内）
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    # 「st.spinner」でグルグル回っている間、表示の不具合が発生しないよう空のエリアを表示
    res_box = st.empty()
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # 初期化状態の確認
            if not hasattr(st.session_state, 'retriever'):
                logger.warning("Retriever not initialized, attempting emergency initialization")
                # 緊急初期化を試行
                try:
                    initialize()
                except Exception as init_error:
                    logger.error(f"Emergency initialization failed: {init_error}")
                    # 完全にフォールバックモードで動作
                    st.session_state.retriever = None
            
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
            
            # 回答が正常に生成されたかチェック
            if not llm_response or 'answer' not in llm_response:
                raise Exception("Invalid response structure generated")
                
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            logger.error(f"Session state debug - retriever: {hasattr(st.session_state, 'retriever')}")
            logger.error(f"Session state debug - mode: {getattr(st.session_state, 'mode', 'UNKNOWN')}")
            
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            
            # 詳細エラー情報（デバッグ用）
            with st.expander("詳細エラー情報（管理者向け）"):
                st.code(f"""
エラー詳細: {str(e)}
初期化状態: {'✓' if hasattr(st.session_state, 'retriever') else '✗'}
Retriever状態: {getattr(st.session_state, 'retriever', 'NOT_SET')}
モード: {getattr(st.session_state, 'mode', 'UNKNOWN')}
""")
            
            # 後続の処理を中断
            st.stop()
    
    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # ==========================================
            # モードが「社内文書検索」の場合
            # ==========================================
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 入力内容と関連性が高い社内文書のありかを表示
                content = cn.display_search_llm_response(llm_response)

            # ==========================================
            # モードが「社内問い合わせ」の場合
            # ==========================================
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # 入力に対しての回答と、参照した文書のありかを表示
                content = cn.display_contact_llm_response(llm_response)
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})