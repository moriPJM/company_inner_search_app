import sys
sys.path.append('.')

# Streamlitセッション状態をモック
class MockSessionState:
    def __init__(self):
        self.data = {}
        self.mode = '社内文書検索'
        self.chat_history = []
        self.retriever = None
        self.vectorstore = None
    
    def __contains__(self, key):
        result = key in self.data or hasattr(self, key)
        print(f'   DEBUG: "{key}" in session_state = {result}')
        return result
        
    def __setattr__(self, name, value):
        if name == 'data':
            object.__setattr__(self, name, value)
        else:
            self.data[name] = value
            object.__setattr__(self, name, value)
            print(f'   DEBUG: セッションに {name} を設定: {value is not None}')
            
    def __getattr__(self, name):
        return self.data.get(name)

import streamlit as st
st.session_state = MockSessionState()

print('=== initialize_retriever内部の動作確認 ===')

try:
    import initialize
    
    # セッション状態にretrieverがあるかチェック
    print('1. 既存retriever確認...')
    has_retriever = 'retriever' in st.session_state
    print(f'   retrieverが既に存在: {has_retriever}')
    
    print('2. initialize_retriever実行...')
    initialize.initialize_retriever()
    
    print('3. 結果確認...')
    print(f'   Retriever設定済み: {st.session_state.retriever is not None}')
    
except Exception as e:
    print(f'エラー: {e}')
    import traceback
    traceback.print_exc()