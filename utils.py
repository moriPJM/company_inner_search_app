"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得（OpenAI APIクォータ制限対策のためモック実装）

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    try:
        # OpenAI LLMを試行
        llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
        
        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
        question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
        question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question_generator_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # モードによってLLMから回答を取得する用のプロンプトを変更
        if st.session_state.mode == ct.ANSWER_MODE_1:
            # モードが「社内文書検索」の場合のプロンプト
            question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
        else:
            # モードが「社内問い合わせ」の場合のプロンプト
            question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
        # LLMから回答を取得する用のプロンプトテンプレートを作成
        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question_answer_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, question_generator_prompt
        )

        # LLMから回答を取得する用のChainを作成
        question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
        # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # LLMへのリクエストとレスポンス取得
        llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        # LLMレスポンスを会話履歴に追加
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=llm_response["answer"])])

        return llm_response
        
    except Exception as e:
        # OpenAI APIクォータ制限の場合、モック回答を返す
        if 'quota' in str(e).lower() or '429' in str(e):
            return get_mock_llm_response(chat_message)
        else:
            raise e


def get_mock_llm_response(chat_message):
    """
    OpenAI APIクォータ制限時のモック回答生成
    
    Args:
        chat_message: ユーザー入力値
        
    Returns:
        モック回答
    """
    # 人事部検索の場合、特別な処理を実行
    if ("人事部" in chat_message or "人事" in chat_message) and ("従業員" in chat_message or "社員" in chat_message or "一覧" in chat_message):
        docs = get_hr_employee_documents(chat_message)
    else:
        # Retrieverから関連ドキュメントを取得
        docs = st.session_state.retriever.invoke(chat_message)
    
    # より詳細なモック回答を生成
    answer = generate_detailed_mock_answer(chat_message, docs)
    
    # 会話履歴に追加
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message), 
        AIMessage(content=answer)
    ])
    
    # モック回答の構築
    mock_response = {
        "input": chat_message,
        "chat_history": st.session_state.chat_history,
        "context": docs,
        "answer": answer
    }
    
    return mock_response


def get_hr_employee_documents(chat_message):
    """
    人事部従業員検索専用のドキュメント取得関数
    
    Args:
        chat_message: ユーザー入力値
        
    Returns:
        関連ドキュメントリスト
    """
    # まず通常のベクトル検索を実行
    docs = st.session_state.retriever.invoke(chat_message)
    
    # 社員名簿が含まれているかチェック
    csv_found = any('社員名簿.csv' in doc.metadata.get('source', '') for doc in docs)
    
    if csv_found:
        return docs
    
    # 社員名簿が見つからない場合、ベクトルストアから直接検索
    if hasattr(st.session_state, 'vectorstore'):
        try:
            # 人事部関連のキーワードで複数回検索
            hr_keywords = ['人事部', '人事', 'HR', '社員名簿', '従業員', '社員']
            all_docs = []
            
            for keyword in hr_keywords:
                keyword_docs = st.session_state.vectorstore.similarity_search(keyword, k=5)
                all_docs.extend(keyword_docs)
            
            # 重複除去
            unique_docs = []
            seen_sources = set()
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                if source not in seen_sources:
                    unique_docs.append(doc)
                    seen_sources.add(source)
            
            # 社員名簿を最優先
            csv_docs = [doc for doc in unique_docs if '社員名簿.csv' in doc.metadata.get('source', '')]
            other_docs = [doc for doc in unique_docs if '社員名簿.csv' not in doc.metadata.get('source', '')]
            
            return csv_docs + other_docs[:4]  # 社員名簿 + その他4つ
            
        except Exception as e:
            print(f"Vectorstore search error: {e}")
    
    return docs  # フォールバック


def generate_detailed_mock_answer(chat_message, docs):
    """
    詳細なモック回答を生成
    
    Args:
        chat_message: ユーザー入力値
        docs: 関連ドキュメント
        
    Returns:
        詳細なモック回答
    """
    try:
        # 議事録ルールに関する質問の場合、特別な回答を生成
        if "議事録" in chat_message and ("ルール" in chat_message or "規則" in chat_message):
            for doc in docs:
                if "議事録ルール.txt" in doc.metadata.get("source", ""):
                    return f"""議事録のルールについてお答えします：

{doc.page_content}

※ 現在OpenAI APIの使用制限のため、簡易版での回答となっています。完全な機能については管理者にお問い合わせください。"""
        
        # 人事部従業員検索の場合、特別な回答を生成
        if ("人事部" in chat_message or "人事" in chat_message) and ("従業員" in chat_message or "社員" in chat_message or "一覧" in chat_message):
            return generate_hr_employee_response(chat_message, docs)
        
        # その他の質問の場合、関連文書の内容を要約して返答
        if docs:
            relevant_content = []
            for i, doc in enumerate(docs[:3]):  # 最初の3つの関連文書を使用
                source = doc.metadata.get("source", "不明")
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                relevant_content.append(f"【関連文書 {i+1}】\n出典: {source}\n内容: {content}")
            
            return f"""「{chat_message}」に関連する情報を検索しました：

{chr(10).join(relevant_content)}

※ 現在OpenAI APIの使用制限のため、関連文書の抜粋のみを表示しています。完全な機能については管理者にお問い合わせください。"""
        
        # 関連文書が見つからない場合
        return f"申し訳ございませんが、「{chat_message}」に関連する文書が見つかりませんでした。現在OpenAI APIの使用制限のため、簡易版での検索となっています。完全な機能については管理者にお問い合わせください。"
    
    except Exception as e:
        # エラーが発生した場合のフォールバック
        return f"回答生成中にエラーが発生しました。申し訳ございませんが、しばらく時間をおいて再度お試しください。エラーの詳細: {str(e)}"


def generate_hr_employee_response(chat_message, docs):
    """
    人事部従業員検索の回答を生成
    
    Args:
        chat_message: ユーザー入力値
        docs: 関連ドキュメント
        
    Returns:
        人事部従業員検索の回答
    """
    try:
        for doc in docs:
            if "社員名簿.csv" in doc.metadata.get("source", ""):
                # 人事部従業員の詳細情報を抽出
                content = doc.page_content
                hr_employees = []
                
                # 人事部従業員を抽出（修正版）
                if "人事部" in content:
                    lines = content.split('\n')
                    current_employee = []
                    collecting_hr_employee = False
                    
                    for line in lines:
                        if "【従業員" in line:
                            # 前の従業員が人事部だった場合、リストに追加
                            if current_employee and collecting_hr_employee:
                                hr_employees.append(current_employee)
                            # 新しい従業員の開始
                            current_employee = [line]
                            collecting_hr_employee = False
                        elif current_employee:
                            current_employee.append(line)
                            # 人事部の記載を発見
                            if "部署: 人事部 (所属部署)" in line:
                                collecting_hr_employee = True
                    
                    # 最後の従業員も処理
                    if current_employee and collecting_hr_employee:
                        hr_employees.append(current_employee)
                
                if hr_employees:
                    # 従業員データを解析して表形式データを作成
                    table_data = []
                    for employee_lines in hr_employees:
                        employee_data = {}
                        for line in employee_lines:
                            if line.startswith('社員ID:'):
                                employee_data['社員ID'] = line.replace('社員ID: ', '').strip()
                            elif line.startswith('氏名（フルネーム）:'):
                                employee_data['氏名'] = line.replace('氏名（フルネーム）: ', '').strip()
                            elif line.startswith('役職:'):
                                employee_data['役職'] = line.replace('役職: ', '').strip()
                            elif line.startswith('メールアドレス:'):
                                employee_data['メールアドレス'] = line.replace('メールアドレス: ', '').strip()
                            elif line.startswith('従業員区分:'):
                                employee_data['従業員区分'] = line.replace('従業員区分: ', '').strip()
                            elif line.startswith('入社日:'):
                                employee_data['入社日'] = line.replace('入社日: ', '').strip()
                            elif line.startswith('年齢:'):
                                employee_data['年齢'] = line.replace('年齢: ', '').strip()
                        
                        # 必要なフィールドが揃っている場合のみ追加
                        if '社員ID' in employee_data and '氏名' in employee_data:
                            table_data.append(employee_data)
                    
                    if table_data:
                        # Streamlitで表を表示するためのコード文字列を含む回答を生成
                        response = f"""人事部に所属している従業員情報を一覧化いたします：

※ 下記の表が表示されない場合は、データの詳細をご確認ください。

**人事部従業員一覧表**

以下の{len(table_data)}名の人事部従業員が在籍しています：

"""
                        # 表形式のデータをMarkdownテーブルとして追加
                        response += "| 社員ID | 氏名 | 役職 | 従業員区分 | 入社日 | メールアドレス |\n"
                        response += "|--------|------|------|------------|--------|----------------|\n"
                        
                        for emp in table_data:
                            emp_id = emp.get('社員ID', '-')
                            name = emp.get('氏名', '-')
                            role = emp.get('役職', '-')
                            category = emp.get('従業員区分', '-')
                            join_date = emp.get('入社日', '-')
                            email = emp.get('メールアドレス', '-')
                            
                            response += f"| {emp_id} | {name} | {role} | {category} | {join_date} | {email} |\n"
                        
                        response += f"\n**人事部従業員総数**: {len(table_data)}名\n\n"
                        response += "※ 現在OpenAI APIの使用制限のため、関連文書の抜粋を表示しています。完全な機能については管理者にお問い合わせください。"
                        
                        # セッション状態に表データを保存（Streamlitで表示用）
                        if hasattr(st.session_state, 'hr_table_data'):
                            st.session_state.hr_table_data = table_data
                        
                        return response
        
        # 人事部従業員が見つからない場合
        return f"申し訳ございませんが、人事部従業員の情報が見つかりませんでした。データの確認をお願いします。"
    
    except Exception as e:
        # エラーが発生した場合のフォールバック
        return f"人事部従業員情報の取得中にエラーが発生しました: {str(e)}"
    
    # その他の質問の場合、関連文書の内容を要約して返答
    if docs:
        relevant_content = []
        for i, doc in enumerate(docs[:3]):  # 最初の3つの関連文書を使用
            source = doc.metadata.get("source", "不明")
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            relevant_content.append(f"【関連文書 {i+1}】\n出典: {source}\n内容: {content}")
        
        return f"""「{chat_message}」に関連する情報を検索しました：

{chr(10).join(relevant_content)}

※ 現在OpenAI APIの使用制限のため、関連文書の抜粋のみを表示しています。完全な機能については管理者にお問い合わせください。"""
    
    # 関連文書が見つからない場合
    return f"申し訳ございませんが、「{chat_message}」に関連する文書が見つかりませんでした。現在OpenAI APIの使用制限のため、簡易版での検索となっています。完全な機能については管理者にお問い合わせください。"