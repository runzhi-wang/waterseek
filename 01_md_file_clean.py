import os
import re
from pathlib import Path
import sys

# 章节标题匹配模式
SECTION_PATTERNS = {
    "introduction": [
        r'^#{1,2}\s*\d*\.?\s*Introduction',
    ],
    "references": [
        r'^#{1,2}\s*References',
    ],
    "acknowledgments": [
        r'^#{1,2}\s*Acknowledg',
    ],
    "credit_author": [
        r'^#{1,2}\s*(Credit author|Author Contribution|CRediT authorship)',
    ],
    "competing_interest": [
        r'^#{1,2}\s*(Declaration|Competing|Conflict|Disclosure)',
    ],
    "appendix_supplementary": [
        r'^#{1,2}\s*(Appendix|Supplementary|Supporting)',
    ],
    "data_availability": [
        r'^#{1,2}\s*Data.*Availability',
        r'^#{1,2}\s*Data availability',
    ],
    "doi_references": [
        r'^#{1,2}\s*DOI.*Reference',
        r'^#{1,2}\s*DOI References',
    ],
}


# 编译正则表达式
def compile_patterns():
    patterns = []
    for patterns_list in SECTION_PATTERNS.values():
        for pattern in patterns_list:
            patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


# 删除章节
def remove_sections(content, patterns):
    lines = content.split('\n')
    output = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检查是否匹配章节标题
        if any(p.match(line.strip()) for p in patterns):
            # 查找当前标题级别
            match = re.match(r'^#+', line.strip())
            if match:
                current_level = len(match.group())

                # 跳过直到遇到同级或更高级标题
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('#'):
                        next_level = len(re.match(r'^#+', next_line).group())
                        if next_level <= current_level:
                            break
                    i += 1
                continue

        output.append(line)
        i += 1

    return '\n'.join(output)


# 处理文件
def process_file(input_path, output_path, patterns):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = remove_sections(content, patterns)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"  处理失败: {e}")
        return False


# 批量处理
def batch_process(input_folder, output_folder, patterns):
    os.makedirs(output_folder, exist_ok=True)
    processed = 0

    for file in Path(input_folder).rglob("*.md"):
        rel_path = file.relative_to(input_folder)
        output_path = Path(output_folder) / rel_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"处理: {file}")
        if process_file(file, output_path, patterns):
            processed += 1

    return processed


# 主函数
def main():
    input_folder = 'E:\\dm\\111'

    output_folder = 'E:\\dm\\md-50-clean'
    patterns = compile_patterns()

    print(f"处理中: {input_folder}")
    print(f"输出到: {output_folder}")

    processed = batch_process(input_folder, output_folder, patterns)

    print(f"完成！已处理 {processed} 个文件")
    print(f"输出目录: {output_folder}")


if __name__ == "__main__":
    main()
