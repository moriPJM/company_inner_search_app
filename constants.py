"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
import pandas as pd

# カスタムCSVローダー関数
def custom_csv_loader(path):
    """
    CSVファイルを1つの統合ドキュメントとして読み込み、検索精度を向上させる
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(path, encoding="utf-8")
        
        # データフレームを整理された文字列形式に変換
        content_parts = []
        
        # ヘッダー情報を追加
        content_parts.append("=== 社員名簿 ===\n")
        content_parts.append(f"総従業員数: {len(df)}名\n")
        
        # 部署別従業員数を追加
        if '部署' in df.columns:
            dept_counts = df['部署'].value_counts()
            content_parts.append("\n=== 部署別従業員数 ===")
            for dept, count in dept_counts.items():
                content_parts.append(f"{dept}: {count}名")
        
        # 各従業員の詳細情報を追加
        content_parts.append("\n\n=== 従業員詳細情報 ===")
        
        for index, row in df.iterrows():
            employee_info = []
            employee_info.append(f"\n【従業員 {index + 1}】")
            
            for column in df.columns:
                value = row[column]
                if pd.notna(value):
                    # 部署情報を強調
                    if column == '部署':
                        employee_info.append(f"{column}: {value} (所属部署)")
                    else:
                        employee_info.append(f"{column}: {value}")
            
            content_parts.append("\n".join(employee_info))
        
        # すべての内容を結合
        full_content = "\n".join(content_parts)
        
        # 検索キーワードの拡張（部署名のバリエーション追加）
        enhanced_content = full_content
        enhanced_content += "\n\n=== 検索用キーワード ===\n"
        enhanced_content += "人事部, 人事, HR, ヒューマンリソース, 人材管理\n"
        enhanced_content += "営業部, 営業, セールス, 売上管理\n"
        enhanced_content += "総務部, 総務, 管理部, 庶務\n"
        enhanced_content += "経理部, 経理, 会計, 財務\n"
        enhanced_content += "IT部, IT, 情報システム, システム管理\n"
        enhanced_content += "マーケティング部, マーケティング, 企画, 宣伝\n"
        
        # Documentオブジェクトを作成
        document = Document(
            page_content=enhanced_content,
            metadata={
                "source": path,
                "file_type": "csv",
                "total_employees": len(df),
                "departments": ", ".join(df['部署'].unique()) if '部署' in df.columns else ""
            }
        )
        
        return [document]
        
    except Exception as e:
        # エラーの場合は標準のCSVLoaderを使用
        print(f"Custom CSV loader error: {e}, falling back to standard CSVLoader")
        loader = CSVLoader(path, encoding="utf-8")
        return loader.load()


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "社内情報特化型生成AI検索アプリ"
ANSWER_MODE_1 = "社内文書検索"
ANSWER_MODE_2 = "社内問い合わせ"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
DOC_SOURCE_ICON = ":material/description: "
LINK_SOURCE_ICON = ":material/link: "
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."


# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"
APP_BOOT_MESSAGE = "アプリが起動されました。"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5


# ==========================================
# RAG参照用のデータソース系
# ==========================================
RAG_TOP_FOLDER_PATH = "./data"
SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": custom_csv_loader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8")
}
WEB_URL_LOAD_TARGETS = [
    "https://generative-ai.web-camp.io/"
]

# チャンク分割設定
CHUNK_SIZE = 500          # チャンクの最大文字数
CHUNK_OVERLAP = 50        # チャンク間の重複文字数
CHUNK_SEPARATOR = "\n"    # チャンク分割の区切り文字


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

SYSTEM_PROMPT_DOC_SEARCH = """
    あなたは社内の文書検索アシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合、空文字「""」を返してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「該当資料なし」と回答してください。

    【文脈】
    {context}
"""

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内情報特化型のアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。
    8. 従業員情報や部署に関する質問の場合、文脈に含まれる社員データから該当する情報を抽出し、表形式で整理して回答してください。
    9. 特定の部署の従業員一覧を求められた場合、該当する全ての従業員の名前、役職、スキル、資格などの情報を含めて回答してください。

    {context}
"""


# ==========================================
# LLMレスポンスの一致判定用
# ==========================================
INQUIRY_NO_MATCH_ANSWER = "回答に必要な情報が見つかりませんでした。"
NO_DOC_MATCH_ANSWER = "該当資料なし"


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
NO_DOC_MATCH_MESSAGE = """
    入力内容と関連する社内文書が見つかりませんでした。\n
    入力内容を変更してください。
"""
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
GET_LLM_RESPONSE_ERROR_MESSAGE = "回答生成に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"