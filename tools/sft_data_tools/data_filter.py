import pandas as pd

# ✅ 定义过滤词列表（支持模糊匹配，不区分大小写）
FILTER_KEYWORDS = [
    # 时间类关键词
    "2013年", "2014年", "2016年", "2017年", "2020", "2021", "2022", "2023", "2007年", "1999年", "1983年", "1970年",
    "2003年", "2004年", "在1965-1966年期间", "2008年", "2015年", "1991年", "2010年", "2011年", "1998年", "1994年",
    "2005年", "1950年",

    # 数据/统计类关键词
    "根据表", "根据给定数据", "年平均", "根据不完全统计", "成果", "公式", "突发环境事件",

    # 文本引用类关键词
    "文中", "根据文本内容", "根据文章内容", "根据相关资料", "根据", "Medvedev", "Nakai", "方程式",

    # 人物/作者类关键词
    "教授", "先生", "作者", "黄芳利", "李世兴", "朱万昌", "Darwin父子", "主编", "1987年", "奥运会", "县村级",

    # 地名/地区类关键词
    "市", "台湾", "国家森林公园", "企业", "学院", "机动车道", "英国", "省", "美国", "蔡邦华", "总经理", "毕业", "三威公司",
    "西峡县", "文献查阅", "的描述", "赵宗哲", "有限公司",

    # 预警相关
    "预警", "农业知识", "What", "模式", "中文名称", "中文名", "编号", "生物学家", "GB", "哪一年", "哪个科", "矿质元素", "哪个属", "学名",
]
# ✅ 输入与输出路径
INPUT_PATH = "/mnt/sda/wyp/forestllm-main/forest_eval/all_class/eval_forest_all_mcq.csv"
OUTPUT_PATH = "/mnt/sda/wyp/forestllm-main/forest_eval/all_class/eval_forest_all_mcq_filtered.csv"

# ✅ 加载数据
df = pd.read_csv(INPUT_PATH)
print(f"原始数据行数: {len(df)}")

# ✅ 预处理：统一成小写关键词
filter_keywords = [kw.lower() for kw in FILTER_KEYWORDS]

# ✅ 判断某一行是否包含过滤词
def contains_filtered_word(row):
    combined_text = " ".join([str(row[col]) for col in ["question", "A", "B", "C", "D"]])
    combined_text = combined_text.lower()
    return any(word in combined_text for word in filter_keywords)

# ✅ 应用过滤逻辑
filtered_df = df[~df.apply(contains_filtered_word, axis=1)].copy()
print(f"过滤后剩余行数: {len(filtered_df)}")

# ✅ 重新编号 id
filtered_df.reset_index(drop=True, inplace=True)
filtered_df["id"] = range(1, len(filtered_df) + 1)


# ✅ 保存
filtered_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"✅ 过滤结果已保存到: {OUTPUT_PATH}")



