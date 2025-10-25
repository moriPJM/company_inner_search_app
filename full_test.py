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
        return key in self.data or hasattr(self, key)
        
    def __setattr__(self, name, value):
        if name == 'data':
            object.__setattr__(self, name, value)
        else:
            self.data[name] = value
            object.__setattr__(self, name, value)
            
    def __getattr__(self, name):
        return self.data.get(name)

import streamlit as st
st.session_state = MockSessionState()

print('=== 完全な機能テスト ===')

# 初期化を実行
print('1. 初期化テスト...')
try:
    import initialize
    initialize.initialize()
    print('   ✓ 初期化成功')
    print(f'   Retriever設定済み: {st.session_state.retriever is not None}')
    print(f'   Vectorstore設定済み: {st.session_state.vectorstore is not None}')
except Exception as e:
    print(f'   ✗ 初期化エラー: {e}')
    import traceback
    traceback.print_exc()

# 回答生成テスト
if hasattr(st.session_state, 'retriever') and st.session_state.retriever is not None:
    print('2. 回答生成テスト...')
    try:
        import utils
        response = utils.get_llm_response('人事部に所属している従業員情報を一覧化して')
        print('   ✓ 回答生成成功')
        answer_preview = response['answer'][:200] + '...'
        print(f'   回答プレビュー: {answer_preview}')
        print(f'   関連ドキュメント数: {len(response.get("context", []))}')
    except Exception as e:
        print(f'   ✗ 回答生成エラー: {e}')
        import traceback
        traceback.print_exc()
else:
    print('2. 回答生成テスト スキップ（retrieverが未設定）')

print('\n=== テスト完了 ===')