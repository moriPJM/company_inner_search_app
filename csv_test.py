import sys
sys.path.append('.')
import constants as ct

# CSVローダーのテスト
csv_path = './data/社員について/社員名簿.csv'
print('=== CSV読み込みテスト ===')

try:
    docs = ct.custom_csv_loader(csv_path)
    print(f'読み込まれたドキュメント数: {len(docs)}')
    
    if docs:
        doc = docs[0]
        print(f'ドキュメント長: {len(doc.page_content)} 文字')
        print(f'メタデータ: {doc.metadata}')
        
        # 人事部の検索をテスト
        content = doc.page_content
        if '人事部' in content:
            print('✓ 人事部の情報が含まれています')
            # 人事部従業員数をカウント
            hr_count = content.count('部署: 人事部')
            print(f'人事部従業員数: {hr_count}名')
            
            # 人事部従業員の詳細表示
            lines = content.split('\n')
            hr_employees = []
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
                    if "部署: 人事部" in line:
                        collecting_hr_employee = True
            
            # 最後の従業員も処理
            if current_employee and collecting_hr_employee:
                hr_employees.append(current_employee)
            
            print(f'抽出された人事部従業員数: {len(hr_employees)}名')
            
            # 最初の人事部従業員の詳細を表示
            if hr_employees:
                print('\n最初の人事部従業員:')
                for line in hr_employees[0]:
                    print(f'  {line}')
        else:
            print('✗ 人事部の情報が見つかりません')
            
except Exception as e:
    print(f'エラー: {e}')
    import traceback
    traceback.print_exc()