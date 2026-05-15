import pandas as pd
import glob
import os

a = 'model-tag-raw-50-1'
path = f"E:\\dm\\{a}\\model-raw"
# 如果文件夹不存在则创建，存在则直接使用
sum_folder = f"E:\\dm\\{a}\\model-sum"
os.makedirs(sum_folder, exist_ok=True)
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
models = [f.replace("md_", "") for f in folders]

# 从all_paper文件获取年份映射
all_paper = pd.read_excel('E:\\test\\large\\all_paper.xlsx')
all_paper['DI_%'] = all_paper['DI'].astype(str).str.replace('/', '%', regex=False)
year_mapping = dict(zip(all_paper['DI_%'], all_paper['PY']))

# 从金标准文件获取source排序参考
gold_standard = pd.read_excel('E:\\dm\\dm-金标准-50.xlsx')
gold_source_order = gold_standard['source'].drop_duplicates().tolist()  # 去重后获取金标准source顺序

for llm in models:
    # 1. 定义文件夹路径
    input_path = f"E:\\dm\\{a}\\model-raw\\md_{llm}"
    output_path = f"E:\\dm\\{a}\\model-sum\\{llm}.xlsx"
    excel_files = glob.glob(os.path.join(input_path, "*.xlsx"))
    data = pd.DataFrame()

    for file in excel_files:
        df = pd.read_excel(file)

        if 'rate_constant' in df.columns:

            filtered_df = df[df['rate_constant'].notna()]
            filtered_df = filtered_df[filtered_df['pollutant_name'].notna()]
            filtered_df = filtered_df[filtered_df['anode'].notna()]

            # 检查哪些source不在金标准中
            if 'source' in filtered_df.columns:

                unmatched_sources = filtered_df.loc[
                    ~filtered_df['source'].isin(gold_source_order),
                    'source'
                ].drop_duplicates()

                if len(unmatched_sources) > 0:

                    print("以下source未在金标准中找到：")

                    for source in unmatched_sources:
                        print(repr(source))

            data = pd.concat([data, filtered_df], ignore_index=True)

    columns_to_drop = ['OEP', 'Rct', 'Rct：', 'pollutant_formula',
                       'pollutant_SMILE']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    if existing_columns:
        data.drop(existing_columns, axis=1, inplace=True)

    rename_dict = {'electrolyte_concentration': 'electrolyte_concentration(mM)',
                   'current_density': 'current_density(mA/cm2)',
                   'temperature': 'temperature(°C)',
                   'solution_volume': 'solution_volume(mL)',
                   'anode_area': 'anode_area(cm2)',
                   'cathode_area': 'cathode_area(cm2)',
                   'electrode_distance': 'electrode_distance(cm)',
                   'rate_constant': 'rate_constant(/min)',
                   'pollutant_concentration': 'pollutant_concentration(mM)'
                   }
    data.rename(columns=rename_dict, inplace=True)

    original_count = len(data)
    data = data.drop_duplicates()
    removed_count = original_count - len(data)
    if removed_count > 0:
        print(f"删除了 {removed_count} 行完全重复的数据")

    # data['year'] = data['source'].map(year_mapping)
    # cols = ['year'] + [col for col in data.columns if col != 'year']
    # data = data[cols]
    columns_to_drop = ['pollutant_category', 'reaction_time', 'removal_rate', 'anode_type']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    if existing_columns:
        data = data.drop(columns=existing_columns)

    for i in range(len(data) - 1):
        aa = data.iloc[i]['source']
        b = data.iloc[i + 1]['source']

        if a.strip() == b.strip() and aa != b:
            print("发现伪相同source")
            print(repr(aa))
            print(repr(b))

    # 插入空行
    if 'source' in data.columns and len(data) > 0:
        rows_to_insert = []
        previous_source = None

        for idx, row in data.iterrows():
            current_source = row['source']
            if previous_source is not None and current_source != previous_source:
                # 插入空行
                rows_to_insert.append(pd.Series([pd.NA] * len(data.columns), index=data.columns))
            rows_to_insert.append(row)
            previous_source = current_source

        data_with_blanks = pd.DataFrame(rows_to_insert, columns=data.columns)
    else:
        data_with_blanks = data

    data_with_blanks.to_excel(output_path, index=False)
    print(f"数据已保存至 {output_path}")