def calculate_average_cer(file_path):
    total_cer = 0.0
    line_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            # 按照格式解析每一行
            parts = line.strip().split(";")
            if len(parts) >= 4:
                cer_part = parts[3].strip()
                if cer_part.startswith("CER: "):
                    cer_value = float(cer_part.split(": ")[1])  # 提取 CER 值
                    total_cer += cer_value
                    line_count += 1

    # 计算平均 CER
    if line_count == 0:
        return 0.0  # 避免除以零
    return total_cer / line_count

if __name__ == "__main__":
    file_path = "../repo_results/numbers/predictions.txt"  # 替换为你的文件路径
    average_cer = calculate_average_cer(file_path)
    print(f"Average CER: {average_cer:.4f}")
