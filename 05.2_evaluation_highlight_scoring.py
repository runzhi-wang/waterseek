import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os


def compare_and_highlight(standard_file_path, model_file_path, output_file_path=None):
    # 排除列
    exclude_columns = ['source', 'Unnamed: 17', 'Unnamed: 18', 'Processing time (s)',
                       'rate_constant_unit', 'processing time (s)']

    # 等价字符串组（且不区分大小写）
    equivalent_string_groups = [
        {'Ti4O7', 'Ti4O7/Ti4O7', 'Ti4O7/null', 'Ti4O7/nan', 'null/Ti4O7', 'nan/Ti4O7', 'Ti4O7无', 'Ti4O7/无',
         'Ti4O7/null'},
        {"Ti/TiSOs", "Ti/Ti4O7"},
        {"porous Ti/BDD", "Ti/BDD"},
        {"Ti/AC", "Ti/activated carbon"},
        {"Ti/BC", "Ti/biochar"},
        {'Ti/Pt/β-PbO2', 'Ti-Pt/β-PbO2'},
        {'Ti/IrO2-Ta2O5', 'Ti/Ta2O5-IrO2'},
        {'copper/null', 'Cu/Cu', 'Cu', 'copper'},
        {'graphite/null', 'graphite/graphite'},
        {'Ti/null', 'Ti/Ti'},
        {'Ti-foam/PbO2', 'Ti-foam/PbO2'},
        {'Ti-foam/PbO2-GN', 'Ti-foam/PbO2-GN'},
        {'activated carbon-Fe2O3-CeO2/Ni,', 'activated carbon modified with Fe2O3-CeO2'},
        {'Ti mesh', 'Ti', 'titanium'},
        {'stainless-steel', 'stainless steel', 'Stainless steel', 'SS304', '304 stainless steel', 'titanium mesh',
         'Ti mesh', 'stainless steel AISI 304'},
        {'BDD', 'BDD/BDD', 'BDD/null', 'null/BDD', 'boron-doped diamond'},
        {'sodium sulfate', 'Na2SO4'},
        {'nickel', 'Ni'},
        {'Perfluorooctanoic acid', 'Perfluorooctanoate'},
        {'4-chlororesorcinol', '4-Chlororesorcinol', 'p-chlororesorcinol', 'p-Chlororesorcinol'},
        {'p-Methoxyphenol', 'P-Methoxyphenol', '4-Methoxyphenol', '4-methoxyphenol'},
        {'p-Nitrophenol', '4-nitrophenol'},
        {'m-Nitrophenol', '3-Nitrophenol'},
        {'o-Nitrophenol', '3-Nitrophenol'},
        {'p-Methoxyphenol', '4-Methoxyphenol'},
        {'planar Ti/BDD', 'Ti/BDD'},
        {'NaH2PO4 + Na2HPO4', 'NaH2PO4/Na2HPO4', 'Na2HPO4 + NaH2PO4', 'Na2HPO4/Na2HPO4'},
        {'Tetracycline hydrochloride', 'Tetracycline'},
        {'NaCl/Na2SO4', 'Na2SO4/NaCl', 'Na2SO4+NaCl', 'NaCl+Na2SO4', 'NaCl + Na2SO4', 'Na2SO4 + NaCl'},
        {'Ti/BDD (NCD)', 'Ti/NCD', 'Ti/BDD'},
        {'Phosphate buffer', 'Na2HPO4/NaH2PO4', 'NaH2PO4/Na2HPO4'},
        {'Ti/Ru30%Ti70%O2', 'Ti/RuO2–TiO2'},
        {'real wastewater', 'urban wastewater'},
        {'carbon fibre', 'null/carbon fibre'},
        {'17beta-Estradiol', '17β-Estradiol'},
        {'Ti-Pt/β-PbO2', 'Ti-Pt/PbO2'},
        {'null/SnOx-La-Sb', 'null/SnOx/La-Sb'},
        {'Ti-Pt/PbO2', 'Ti-Pt/β-PbO2'},
        {'25/20', '25', '20'},
        {'gold', 'Au'}
    ]

    if output_file_path is None:
        model_dir = os.path.dirname(model_file_path)
        model_name = os.path.basename(model_file_path)
        name, ext = os.path.splitext(model_name)
        output_file_path = os.path.join(model_dir, f"{name}_color{ext}")

    # 加载数据
    standard_df = pd.read_excel(standard_file_path)
    model_df = pd.read_excel(model_file_path)
    model_wb = load_workbook(model_file_path)
    model_ws = model_wb['Sheet1']

    # 定义颜色填充
    green = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")  # 正确
    blue = PatternFill(start_color="0070C0", end_color="0070C0", fill_type="solid")  # 错误
    purple = PatternFill(start_color="7030A0", end_color="7030A0", fill_type="solid")  # 新增行
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # NaN 匹配

    # 按 source 统计标准 vs 模型行数
    standard_source_counts = standard_df.groupby('source').size()
    model_source_counts = model_df.groupby('source').size()
    insertions = []

    # 处理所有标准文件中的DOI
    for source in standard_source_counts.index:
        if source in model_source_counts.index:
            # 部分缺失：DOI存在但行数不足
            standard_count = standard_source_counts[source]
            model_count = model_source_counts[source]

            if model_count < standard_count:
                rows_to_insert = standard_count - model_count
                # 找到该source在model文件中的最后一行位置
                source_indices = model_df[model_df['source'] == source].index.tolist()
                if source_indices:
                    # 该source的最后一行在Excel中的行号，插入位置是最后一行之后
                    last_row_excel = source_indices[-1] + 2
                    insert_position = last_row_excel + 1  # 在最后一行之后插入
                    insertions.append((source, insert_position, rows_to_insert))
        else:
            # 完全缺失：DOI在模型文件中完全不存在
            standard_count = standard_source_counts[source]
            rows_to_insert = standard_count
            # 插入到文件末尾
            insert_position = model_ws.max_row + 1
            insertions.append((source, insert_position, rows_to_insert))

    # 按插入位置从大到小排序（从文件末尾开始处理）
    insertions.sort(key=lambda x: x[1], reverse=True)
    for source, insert_position, rows_to_insert in insertions:
        # 在指定位置插入行
        for i in range(rows_to_insert):
            model_ws.insert_rows(insert_position)
            # 在新插入行的source列填入DOI
            source_col_idx = list(model_df.columns).index('source') + 1
            model_ws.cell(row=insert_position, column=source_col_idx).value = source
            # 将新插入的行填充为紫色
            for col in range(1, len(model_df.columns) + 1):
                col_name = list(model_df.columns)[col - 1]
                if col_name not in exclude_columns:
                    model_ws.cell(row=insert_position, column=col).fill = purple

    # 保存并重新加载数据
    if insertions:  # 只有在有插入操作时才需要重新加载
        model_wb.save(output_file_path)
        model_df = pd.read_excel(output_file_path)
        model_wb = load_workbook(output_file_path)
        model_ws = model_wb['Sheet1']

    # 获取所有列名
    columns = list(model_df.columns)

    # 创建匹配键的函数
    def create_match_key(row):
        source = str(row['source']) if pd.notna(row['source']) else ""
        rate_constant = str(row['rate_constant(/min)']) if pd.notna(row['rate_constant(/min)']) else ""
        try:
            # 将k值转换为浮点数进行四舍五入，确保精度一致
            if rate_constant:
                k_float = float(rate_constant)
                # 使用格式化字符串避免科学计数法
                rate_constant = f"{k_float:.6f}"
        except (ValueError, TypeError):
            pass
        return f"{source}_{rate_constant}"

    # 检查两个k值是否匹配（相对误差1%以内）
    def k_values_match(k1, k2):
        try:
            k1_float = float(k1) if pd.notna(k1) else None
            k2_float = float(k2) if pd.notna(k2) else None

            if k1_float is None or k2_float is None:
                return False

            # 处理接近零的情况
            if abs(k1_float) < 1e-10 and abs(k2_float) < 1e-10:
                return True

            # 计算相对误差
            relative_error = abs(k1_float - k2_float) / abs(k2_float)

            # 相对误差在1%以内视为匹配
            return relative_error <= 0.01
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    # 检查两个字符串是否相等的函数（支持大小写不敏感和等价组）
    def strings_equal(str1, str2):
        if pd.isna(str1) or str(str1).strip() in ['', 'NaN', 'None']:
            str1 = ''
        if pd.isna(str2) or str(str2).strip() in ['', 'NaN', 'None']:
            str2 = ''

        str1_clean = str(str1).strip().lower()
        str2_clean = str(str2).strip().lower()

        if str1_clean == str2_clean:
            return True

        for group in equivalent_string_groups:
            group_lower = {s.lower() for s in group}
            if str1_clean in group_lower and str2_clean in group_lower:
                return True

        return False

    # 构建标准字典
    standard_dict = {}
    for _, row in standard_df.iterrows():
        key = create_match_key(row)
        standard_dict[key] = row

    # 同时构建一个更宽松的匹配字典，用于处理k值近似的情况
    loose_standard_dict = {}
    for _, row in standard_df.iterrows():
        source = str(row['source']) if pd.notna(row['source']) else ""
        k_value = row['rate_constant(/min)'] if pd.notna(row['rate_constant(/min)']) else None
        loose_key = f"{source}_k"
        if source not in loose_standard_dict:
            loose_standard_dict[source] = {}
        loose_standard_dict[source][k_value] = row

    matched_count = 0
    not_matched_count = 0
    loosely_matched_count = 0
    inserted_rows_count = 0  # 新增行计数器

    for idx, model_row in model_df.iterrows():
        row_idx_excel = idx + 2  # Excel 行号

        # 检查当前行是否是新插入的行（紫色行）
        is_inserted_row = False
        for col in range(1, len(columns) + 1):
            cell = model_ws.cell(row=row_idx_excel, column=col)
            if cell.fill.start_color.index == purple.start_color.index:
                is_inserted_row = True
                break

        # 如果是新插入的行，跳过匹配逻辑
        if is_inserted_row:
            inserted_rows_count += 1
            continue

        model_key = create_match_key(model_row)
        model_source = str(model_row['source']) if pd.notna(model_row['source']) else ""
        model_k = model_row['rate_constant(/min)']

        if model_key in standard_dict:
            matched_count += 1
            standard_row = standard_dict[model_key]
            standard_k = standard_row['rate_constant(/min)']

            for col_name in columns:
                col_idx = columns.index(col_name) + 1  # Excel列号

                if col_name in exclude_columns:
                    continue

                model_val = model_row[col_name]
                std_val = standard_row[col_name] if col_name in standard_row else None

                cell = model_ws.cell(row=row_idx_excel, column=col_idx)

                if pd.isna(model_val) and (pd.isna(std_val) or std_val is None):
                    cell.fill = yellow
                elif pd.isna(model_val) and not pd.isna(std_val):
                    cell.fill = purple
                elif not pd.isna(model_val) and pd.isna(std_val):
                    cell.fill = blue
                else:
                    # 数值比较
                    try:
                        m_float = float(model_val) if not pd.isna(model_val) else None
                        s_float = float(std_val) if not pd.isna(std_val) else None

                        if m_float is not None and s_float is not None:
                            if col_name == 'rate_constant(/min)':
                                # 对k值使用1%的相对误差容忍度
                                if abs(m_float - s_float) / abs(s_float) <= 0.01:
                                    cell.fill = green
                                else:
                                    cell.fill = blue
                            else:
                                # 其他数值列使用原来的5%
                                if abs(m_float - s_float) / abs(s_float) <= 0.05:
                                    cell.fill = green
                                else:
                                    cell.fill = blue
                        else:
                            if strings_equal(model_val, std_val):
                                cell.fill = green
                            else:
                                cell.fill = blue
                    except (ValueError, TypeError, ZeroDivisionError):
                        if strings_equal(model_val, std_val):
                            cell.fill = green
                        else:
                            cell.fill = blue
        else:
            # 尝试宽松匹配：source相同，k值在1%误差范围内
            found_loose_match = False
            if model_source in loose_standard_dict:
                for std_k, standard_row in loose_standard_dict[model_source].items():
                    if k_values_match(model_k, std_k):
                        loosely_matched_count += 1
                        found_loose_match = True

                        for col_name in columns:
                            col_idx = columns.index(col_name) + 1  # Excel列号

                            if col_name in exclude_columns:
                                continue

                            model_val = model_row[col_name]
                            std_val = standard_row[col_name] if col_name in standard_row else None

                            cell = model_ws.cell(row=row_idx_excel, column=col_idx)

                            if pd.isna(model_val) and (pd.isna(std_val) or std_val is None):
                                cell.fill = yellow
                            elif pd.isna(model_val) and not pd.isna(std_val):
                                cell.fill = purple
                            elif not pd.isna(model_val) and pd.isna(std_val):
                                cell.fill = blue
                            else:
                                try:
                                    m_float = float(model_val) if not pd.isna(model_val) else None
                                    s_float = float(std_val) if not pd.isna(std_val) else None

                                    if m_float is not None and s_float is not None:
                                        if col_name == 'rate_constant(/min)':
                                            # k值已经匹配，标绿色
                                            cell.fill = green
                                        else:
                                            if abs(m_float - s_float) / abs(s_float) <= 0.05:
                                                cell.fill = green
                                            else:
                                                cell.fill = blue
                                    else:
                                        if strings_equal(model_val, std_val):
                                            cell.fill = green
                                        else:
                                            cell.fill = blue
                                except (ValueError, TypeError, ZeroDivisionError):
                                    if strings_equal(model_val, std_val):
                                        cell.fill = green
                                    else:
                                        cell.fill = blue
                        break

            if not found_loose_match:
                not_matched_count += 1
                # 标记为蓝色
                for col_name in columns:
                    col_idx = columns.index(col_name) + 1
                    if col_name not in exclude_columns:
                        cell = model_ws.cell(row=row_idx_excel, column=col_idx)
                        cell.fill = blue

    # 处理source列为空的行，设置填充色为无
    for idx, model_row in model_df.iterrows():
        row_idx_excel = idx + 2
        source_value = model_row['source']

        # 检查source列是否为空
        if pd.isna(source_value) or str(source_value).strip() == '':
            # 将该行的所有单元格填充色设置为无
            for col in range(1, len(model_df.columns) + 1):
                col_name = list(model_df.columns)[col - 1]
                if col_name not in exclude_columns:
                    cell = model_ws.cell(row=row_idx_excel, column=col)
                    # 设置填充为无（白色）
                    cell.fill = PatternFill(fill_type=None)

    # 保存结果
    model_wb.save(output_file_path)

    print(f"匹配情况统计:")
    print(f"- 精确匹配: {matched_count} 行")
    print(f"- 宽松匹配(k值1%误差内): {loosely_matched_count} 行")
    print(f"- 未匹配: {not_matched_count} 行")
    print(f"- 新插入行: {inserted_rows_count} 行")
    print(f"结果已保存到: {output_file_path}")


def debug_matching(standard_file_path, model_file_path):
    """调试函数，用于检查匹配情况"""

    standard_df = pd.read_excel(standard_file_path, sheet_name='Sheet1')
    model_df = pd.read_excel(model_file_path, sheet_name='Sheet1')

    print(f"标准文件记录数: {len(standard_df)}")
    print(f"模型文件记录数: {len(model_df)}")

    # 显示每个source的行数统计
    standard_source_counts = standard_df.groupby('source').size()
    model_source_counts = model_df.groupby('source').size()

    for source in standard_source_counts.index:
        std_count = standard_source_counts[source]
        model_count = model_source_counts.get(source, 0)
        diff = std_count - model_count
        status = "匹配" if diff == 0 else f"缺少{diff}行" if diff > 0 else f"多出{-diff}行"
        print(f"{source}: 标准{std_count}行, 模型{model_count}行 - {status}")


def main():
    standard_file_path = "E:\\dm\\dm-金标准-50.xlsx"
    a = 'raw-50-1'
    path = f"E:\\dm\\model-tag-{a}\\model-raw"
    # 输出文件夹,如果不存在则自动创建
    folder = f"E:\\dm\\model-tag-{a}\\model-color"
    os.makedirs(folder, exist_ok=True)
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    models = [f.replace("md_", "") for f in folders]
    for model in models:
        model_file_path = f"E:\\dm\\model-tag-{a}\\model-sum\\{model}.xlsx"
        folder = f"E:\\dm\\model-tag-{a}\\model-color"
        output_path = os.path.join(folder, f"{model}.xlsx")
        debug_matching(standard_file_path, model_file_path)
        compare_and_highlight(standard_file_path, model_file_path, output_path)


if __name__ == "__main__":
    main()
