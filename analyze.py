import pandas as pd
from typing import List, Optional


def save_floats_to_excel(
        float_list: List[float],
        excel_filename: str = "float_data.xlsx",
        sheet_name: str = "FloatData"
) -> None:
    """
    将浮点数列表保存到 Excel 文件中

    参数:
        float_list: 包含浮点数的列表（支持 int 类型，会自动转换为 float）
        excel_filename: 输出的 Excel 文件名（默认：float_data.xlsx）
        sheet_name: Excel 工作表名称（默认：FloatData）

    异常:
        TypeError: 输入不是列表，或列表中包含非数字类型元素
        ValueError: 输入列表为空
        PermissionError: 无法写入文件（文件被占用等）
        Exception: 其他未知错误
    """
    # 1. 数据验证：检查是否为列表
    if not isinstance(float_list, list):
        raise TypeError("输入必须是一个列表")

    # 2. 检查列表是否为空
    if len(float_list) == 0:
        raise ValueError("输入列表不能为空")

    # 3. 验证并转换列表元素为 float（支持 int 类型输入）
    try:
        # 尝试将所有元素转换为 float，过滤非数字类型
        validated_data = [float(item) for item in float_list]
    except (TypeError, ValueError):
        raise TypeError("列表中包含非数字类型元素，请确保所有元素都是浮点数或整数")

    # 4. 转换为 DataFrame（方便保存为 Excel，且格式更规范）
    # 创建单列 DataFrame，列名设为 "FloatValues"
    df = pd.DataFrame(validated_data, columns=["FloatValues"])

    # 5. 保存到 Excel 文件
    try:
        # 使用 pandas 的 to_excel 方法，index=False 表示不保存行索引
        df.to_excel(excel_filename, sheet_name=sheet_name, index=False)
        print(f"✅ 数据已成功保存到：{excel_filename}（工作表：{sheet_name}）")
        print(f"📊 共保存 {len(validated_data)} 个浮点数")
    except PermissionError:
        raise PermissionError(f"无法写入文件 {excel_filename}，可能是文件已被打开或没有写入权限")
    except Exception as e:
        raise Exception(f"保存 Excel 时发生错误：{str(e)}")


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 示例 1：基本使用（默认文件名和工作表名）
    data1 = [1.23, 4.56, 7.89, 0.10, 3.1415926]
    save_floats_to_excel(data1)

    # 示例 2：自定义文件名和工作表名
    data2 = [2.46, 8.10, 12.14, 16.18]
    save_floats_to_excel(
        float_list=data2,
        excel_filename="custom_float_data.xlsx",
        sheet_name="MyData"
    )

    # 示例 3：支持整数（会自动转换为浮点数）
    data3 = [1, 3, 5, 7, 9]  # 最终保存为 [1.0, 3.0, 5.0, 7.0, 9.0]
    save_floats_to_excel(data3, excel_filename="int_to_float.xlsx")