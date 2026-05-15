import requests
import pandas as pd
import os
from urllib.parse import quote
import time
import json


def get_compound_info(molecule_name):
    """获取化合物的CID、分子式和SMILES"""
    try:
        # 第一步：获取CID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(molecule_name)}/cids/JSON"
        cid_response = requests.get(search_url, timeout=30)
        cid_data = cid_response.json()

        cid = cid_data['IdentifierList']['CID'][0] if cid_data['IdentifierList']['CID'] else None
        if not cid:
            return {'status': '未找到CID'}

        # 第二步：获取详细属性
        property_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,SMILES,CanonicalSMILES,IsomericSMILES,IUPACName/JSON"
        prop_response = requests.get(property_url, timeout=30)
        prop_data = prop_response.json()

        props = prop_data['PropertyTable']['Properties'][0]
        smiles = props.get('CanonicalSMILES') or props.get('SMILES') or props.get('IsomericSMILES', '')

        return {
            'cid': cid,
            'formula': props.get('MolecularFormula', ''),
            'smiles': smiles,
            'iupac_name': props.get('IUPACName', ''),
            'status': '成功'
        }

    except Exception as e:
        return {'status': f'错误: {str(e)}'}


def query_compounds(molecule_list, output_file, start_index=0, save_interval=10):
    """
    批量查询化合物信息

    参数:
        molecule_list: 化合物名称列表
        output_file: 输出文件路径
        start_index: 从第几个分子开始（从0开始计数）
        save_interval: 每完成多少个保存一次
    """
    # 检查是否存在进度文件
    progress_file = output_file.replace('.xlsx', '_progress.json')
    results = []

    # 如果存在进度文件，读取已处理的结果
    if os.path.exists(output_file) and os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        existing_df = pd.read_excel(output_file)
        results = existing_df.to_dict('records')

        print(f"检测到进度文件，已处理 {progress['processed_count']} 个分子")
        print(f"上次处理到: {progress['last_processed']}")

        # 如果指定了起始索引，使用指定值，否则使用保存的进度
        if start_index == 0:
            start_index = progress['processed_count']

    total_molecules = len(molecule_list)

    for i in range(start_index, total_molecules):
        mol = molecule_list[i]
        current_number = i + 1
        print(f"[{current_number}/{total_molecules}] 处理: {mol}")

        result = get_compound_info(mol)
        result_row = {
            'name': mol,
            'formula': result.get('formula', ''),
            'smiles': result.get('smiles', ''),
            'cid': result.get('cid', ''),
            'iupac_name': result.get('iupac_name', ''),
            'status': result.get('status', '未知错误')
        }
        results.append(result_row)

        if result.get('status') == '成功':
            print(f"  分子式: {result.get('formula')}")
            print(f"  SMILES: {result.get('smiles')[:50]}..." if len(
                result.get('smiles', '')) > 50 else f"  SMILES: {result.get('smiles')}")
        else:
            print(f"  失败: {result.get('status')}")

        # 每 save_interval 个分子保存一次
        if current_number % save_interval == 0 or current_number == total_molecules:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)

            progress_data = {
                'processed_count': current_number,
                'last_processed': mol,
                'last_saved_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            print(f"  已保存进度 (处理到第 {current_number} 个)")

        time.sleep(1)

    print(f"处理完成! 共处理 {total_molecules} 个分子")
    print(f"结果已保存到: {output_file}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    excel_path = 'E:\\desktop\\op.xlsx'

    start_choice = input("是否从指定位置开始处理？(y/n, 默认从头开始): ")
    if start_choice.lower() == 'y':
        try:
            start_index = int(input("请输入起始索引（从0开始）: "))
        except:
            print("输入无效，将从0开始")
            start_index = 0
    else:
        start_index = 0

    save_interval = 10
    save_choice = input("是否修改保存间隔？(y/n, 默认10个保存一次): ")
    if save_choice.lower() == 'y':
        try:
            save_interval = int(input("请输入保存间隔: "))
        except:
            pass

    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 查找化合物名称列
    name_columns = ['name', 'Name', 'compound', 'Compound']
    name_col = None

    for col in name_columns:
        if col in df.columns:
            name_col = col
            break

    if name_col is None:
        name_col = df.columns[0]
        print(f"未找到标准名称列，使用第一列: {name_col}")

    molecules = df[name_col].dropna().astype(str).tolist()

    print(f"从文件读取到 {len(molecules)} 个化合物名称")
    print(f"从第 {start_index + 1} 个分子开始处理")
    print(f"每 {save_interval} 个分子保存一次进度")

    # 执行查询
    results_df = query_compounds(
        molecule_list=molecules,
        output_file=excel_path,
        start_index=start_index,
        save_interval=save_interval
    )

    if results_df is not None:
        print(f"查询结果统计:")
        print(f"  成功: {len(results_df[results_df['status'] == '成功'])}")
        print(f"  失败: {len(results_df[results_df['status'] != '成功'])}")