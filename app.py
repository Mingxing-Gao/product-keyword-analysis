import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 初始化页面
st.set_page_config(page_title="通用产品关键词分析工具", layout="wide")

# 下载必要的NLTK资源
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# 页面标题
st.title("通用产品关键词分析工具")
st.write("上传任何产品的Excel数据，自动分析关键词及评论量统计")

# 文件上传
uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"])
core_keyword = st.text_input("输入核心关键词 (例如: cat bed, yoga pants, coffee maker 等)")

# 定义通用产品特性分类器
def classify_keyword(keyword, core_terms):
    """根据关键词特征将其分类为产品类型、功能特性、材质特征或适用对象"""
    
    # 预定义特征词汇集合
    function_terms = {
        'high', 'waist', 'pocket', 'pockets', 'control', 'tummy', 'compression', 
        'anti', 'slip', 'removable', 'washable', 'machine', 'waterproof', 'proof',
        'adjustable', 'portable', 'foldable', 'collapsible', 'cooling', 'warming',
        'calming', 'anxiety', 'orthopedic', 'support', 'breathable', 'non-slip',
        'nonslip', 'non-skid', 'nonskid', 'zipper', 'button', 'stretch', 'stretchable',
        'seamless'
    }
    
    material_terms = {
        'cotton', 'polyester', 'nylon', 'spandex', 'leather', 'plastic', 'metal',
        'wood', 'silicone', 'glass', 'ceramic', 'plush', 'soft', 'fluffy', 'fur',
        'fleece', 'sherpa', 'mesh', 'wool', 'velvet', 'microfiber', 'foam',
        'memory', 'rubber', 'fabric', 'buttery', 'luxe', 'silk', 'satin',
        'stainless', 'steel', 'aluminum', 'bamboo', 'denim', 'linen', 'corduroy'
    }
    
    target_terms = {
        'women', 'men', 'kids', 'children', 'baby', 'adult', 'senior', 'indoor',
        'outdoor', 'travel', 'home', 'office', 'kitchen', 'bathroom', 'bedroom',
        'living', 'car', 'small', 'medium', 'large', 'extra', 'petite', 'tall',
        'plus', 'size', 'regular', 'professional', 'beginner', 'advanced', 'casual'
    }
    
    # 检查关键词是否包含核心词，如果包含，则可能是产品类型
    for core in core_terms:
        if core in keyword and len(keyword) > len(core):
            return "产品类型"
    
    # 检查关键词是否属于其他类别
    keyword_parts = keyword.lower().split()
    
    for word in keyword_parts:
        if word in function_terms:
            return "功能特性"
        if word in material_terms:
            return "材质特征"
        if word in target_terms:
            return "适用对象"
    
    # 默认情况下，如果关键词与核心词完全匹配，归类为产品类型
    if keyword.lower() in [term.lower() for term in core_terms]:
        return "产品类型"
        
    # 如果无法分类，则根据词长度进行简单猜测
    if len(keyword) < 4:
        return "其他"
    elif any(word in material_terms for word in keyword_parts):
        return "材质特征"
    elif any(word in function_terms for word in keyword_parts):
        return "功能特性"
    else:
        return "其他"

# 提取关键词
def extract_keywords(title, core_keyword):
    """从标题中提取关键词"""
    # 将核心关键词分解为单词
    core_terms = core_keyword.lower().split()
    
    # 对标题进行词元化
    title = title.lower()
    tokens = word_tokenize(title)
    
    # 获取英文停用词
    stop_words = set(stopwords.words('english'))
    
    # 过滤掉短词和停用词
    filtered_tokens = [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]
    
    # 提取2-3个词的短语（可能是重要特征）
    phrases = []
    for i in range(len(tokens) - 1):
        if tokens[i].isalpha() and tokens[i+1].isalpha():
            if tokens[i] not in stop_words or tokens[i+1] not in stop_words:
                phrases.append(f"{tokens[i]} {tokens[i+1]}")
    
    for i in range(len(tokens) - 2):
        if tokens[i].isalpha() and tokens[i+1].isalpha() and tokens[i+2].isalpha():
            if tokens[i] not in stop_words or tokens[i+1] not in stop_words or tokens[i+2] not in stop_words:
                phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    
    # 组合单词和短语
    all_keywords = filtered_tokens + phrases
    
    # 对关键词进行分类
    classified_keywords = {}
    for keyword in all_keywords:
        category = classify_keyword(keyword, core_terms)
        if category not in classified_keywords:
            classified_keywords[category] = []
        classified_keywords[category].append(keyword)
    
    return classified_keywords

# 统计关键词频率
def count_keywords_by_reviews(df, title_column, review_column, core_keyword):
    """统计关键词出现频率并按评论数加权"""
    category_stats = {
        "产品类型": {},
        "功能特性": {},
        "材质特征": {},
        "适用对象": {},
        "其他": {}
    }
    
    # 确保评论列是数值类型
    df[review_column] = pd.to_numeric(df[review_column], errors='coerce').fillna(0)
    
    # 处理每一行
    for _, row in df.iterrows():
        title = str(row[title_column])
        review_count = row[review_column]
        
        # 提取并分类关键词
        classified_keywords = extract_keywords(title, core_keyword)
        
        # 统计每个分类的关键词
        for category, keywords in classified_keywords.items():
            # 去重关键词，确保同一标题中相同关键词只统计一次
            unique_keywords = set(keywords)
            for keyword in unique_keywords:
                if keyword not in category_stats[category]:
                    category_stats[category][keyword] = review_count
                else:
                    category_stats[category][keyword] += review_count
    
    # 排序每个分类中的关键词，按评论数从高到低
    for category in category_stats:
        category_stats[category] = {k: v for k, v in sorted(
            category_stats[category].items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
    
    return category_stats

# 翻译常用词
def translate_common_terms(term):
    """将常见英文词转为中文，用于展示"""
    translations = {
        # 产品类型词
        "bed": "床",
        "cave": "洞/窝洞",
        "house": "屋",
        "tent": "帐篷",
        "sofa": "沙发",
        "couch": "长椅",
        "mat": "垫子",
        "pants": "裤子",
        "leggings": "紧身裤",
        "yoga": "瑜伽",
        "flare": "喇叭裤",
        "bootcut": "靴型裤",
        "wide leg": "阔腿裤",
        "capri": "七分裤",
        "shorts": "短裤",
        
        # 功能特性词
        "high waist": "高腰",
        "waisted": "高腰",
        "pockets": "带口袋",
        "pocket": "带口袋",
        "tummy control": "收腹",
        "control": "控制型",
        "compression": "压缩",
        "washable": "可洗",
        "machine": "机洗",
        "waterproof": "防水",
        "slip": "防滑",
        "removable": "可拆卸",
        "foldable": "可折叠",
        "calming": "舒缓",
        "anxiety": "防焦虑",
        "orthopedic": "骨科",
        
        # 材质特征词
        "soft": "柔软",
        "plush": "毛绒",
        "fluffy": "蓬松",
        "fur": "皮毛",
        "fleece": "摇粒绒",
        "sherpa": "绵羊绒",
        "buttery": "如黄油般柔软",
        "cotton": "棉质",
        "polyester": "聚酯",
        "spandex": "氨纶",
        "nylon": "尼龙",
        
        # 适用对象词
        "indoor": "室内",
        "small": "小型",
        "medium": "中型",
        "large": "大型",
        "women": "女士",
        "men": "男士",
        "kids": "儿童",
        "puppy": "幼犬",
        "kitten": "幼猫",
        "petite": "矮个",
        "tall": "高个",
        "plus size": "大码"
    }
    
    if term.lower() in translations:
        return translations[term.lower()]
    else:
        return ""  # 如果没有翻译，返回空字符串

# 生成描述
def generate_description(term, category):
    """根据词语和类别生成描述"""
    descriptions = {
        # 产品类型描述
        "产品类型": {
            "bed": "基础床型设计",
            "cave": "半封闭式设计",
            "house": "结构化设计",
            "tent": "帐篷式设计",
            "sofa": "沙发式设计",
            "couch": "长椅式设计",
            "mat": "薄垫式设计",
            "pants": "基础裤型",
            "leggings": "贴身弹性裤",
            "yoga": "瑜伽专用",
            "flare": "下摆喇叭形状",
            "bootcut": "微喇设计",
            "wide leg": "宽松腿型",
            "capri": "中长款",
            "shorts": "短款设计"
        },
        
        # 功能特性描述
        "功能特性": {
            "high waist": "高腰设计",
            "waisted": "高腰设计",
            "pockets": "实用口袋设计",
            "pocket": "实用口袋设计",
            "tummy control": "塑形收腹功能",
            "control": "提供塑型控制",
            "compression": "肌肉支撑压缩",
            "washable": "可清洗的设计",
            "machine": "可机器清洗",
            "waterproof": "防水功能设计",
            "slip": "底部防滑设计",
            "removable": "可拆卸清洗的设计",
            "foldable": "可折叠存储的设计",
            "calming": "减轻焦虑的设计",
            "anxiety": "缓解焦虑感的设计",
            "orthopedic": "提供关节支撑功能"
        },
        
        # 材质特征描述
        "材质特征": {
            "soft": "柔软舒适的材质",
            "plush": "绒毛材质",
            "fluffy": "松软的表面材质",
            "fur": "皮毛材质",
            "fleece": "摇粒绒材质",
            "sherpa": "绵羊绒材质",
            "buttery": "超柔软手感",
            "cotton": "棉质材料",
            "polyester": "聚酯材质",
            "spandex": "含氨纶弹性材质",
            "nylon": "含尼龙材质"
        },
        
        # 适用对象描述
        "适用对象": {
            "indoor": "适合室内使用",
            "small": "适合小型身材",
            "medium": "适合中型身材",
            "large": "适合大型身材",
            "women": "适合女性",
            "men": "适合男性",
            "kids": "适合儿童",
            "puppy": "适合幼犬",
            "kitten": "适合幼猫",
            "petite": "适合矮个身材",
            "tall": "适合高个身材",
            "plus size": "适合大码身材"
        }
    }
    
    if category in descriptions and term.lower() in descriptions[category]:
        return descriptions[category][term.lower()]
    else:
        # 如果没有预定义描述，生成通用描述
        if category == "产品类型":
            return f"{term}类型产品"
        elif category == "功能特性":
            return f"提供{term}功能"
        elif category == "材质特征":
            return f"{term}材质"
        elif category == "适用对象":
            return f"适合{term}使用"
        else:
            return ""

# 主程序逻辑
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    columns = df.columns.tolist()
    
    title_column = st.selectbox("选择标题列", columns)
    review_column = st.selectbox("选择评论数量列", columns)
    
    if st.button("分析"):
        if not core_keyword:
            st.error("请输入核心关键词")
        else:
            with st.spinner("正在分析关键词..."):
                # 进行关键词统计
                keyword_stats = count_keywords_by_reviews(df, title_column, review_column, core_keyword)
                
                # 显示分析结果
                st.subheader("关键词分析结果")
                
                # 创建结果表格
                for category, keywords in keyword_stats.items():
                    if keywords:  # 只显示有内容的分类
                        st.write(f"### {category}关键词统计")
                        
                        # 创建结果数据框
                        result_data = []
                        for kw, count in list(keywords.items())[:20]:  # 只取前20个结果避免过长
                            chinese_translation = translate_common_terms(kw)
                            description = generate_description(kw, category)
                            result_data.append({
                                "英文关键词": kw,
                                "中文对应词": chinese_translation,
                                f"{category}描述": description,
                                "评论数累计": int(count)
                            })
                        
                        if result_data:
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df)
                            
                            # 生成可视化图表
                            if len(result_data) > 0:
                                top_n = min(10, len(result_data))
                                fig, ax = plt.subplots(figsize=(10, 6))
                                chart_data = result_df.head(top_n)
                                sns.barplot(x="评论数累计", y="英文关键词", data=chart_data, ax=ax)
                                ax.set_title(f"Top {top_n} {category}关键词评论数")
                                st.pyplot(fig)
                
                # 组合关键词分析
                st.subheader("高评论量关键词组合")
                
                # 从每个类别中获取前2个关键词
                top_product_types = list(keyword_stats["产品类型"].keys())[:2] if keyword_stats["产品类型"] else []
                top_features = list(keyword_stats["功能特性"].keys())[:2] if keyword_stats["功能特性"] else []
                top_materials = list(keyword_stats["材质特征"].keys())[:2] if keyword_stats["材质特征"] else []
                top_targets = list(keyword_stats["适用对象"].keys())[:2] if keyword_stats["适用对象"] else []
                
                # 创建组合
                combinations = []
                
                # 产品类型 + 功能特性
                for p in top_product_types:
                    for f in top_features:
                        combinations.append((p, f))
                
                # 产品类型 + 材质
                for p in top_product_types:
                    for m in top_materials:
                        combinations.append((p, m))
                
                # 产品类型 + 适用对象
                for p in top_product_types:
                    for t in top_targets:
                        combinations.append((p, t))
                
                # 功能特性 + 材质
                for f in top_features:
                    for m in top_materials:
                        combinations.append((f, m))
                
                # 计算组合的评论数累计
                combo_results = []
                for combo in combinations:
                    word1, word2 = combo
                    
                    # 获取各个词的评论数
                    word1_count = 0
                    word2_count = 0
                    
                    for category in keyword_stats:
                        if word1 in keyword_stats[category]:
                            word1_count = keyword_stats[category][word1]
                        if word2 in keyword_stats[category]:
                            word2_count = keyword_stats[category][word2]
                    
                    # 计算中文翻译
                    word1_chinese = translate_common_terms(word1)
                    word2_chinese = translate_common_terms(word2)
                    
                    combo_results.append({
                        "英文组合关键词": f"{word1} {word2}",
                        "中文对应词": f"{word1_chinese}{word2_chinese}",
                        "评论数累计": word1_count + word2_count
                    })
                
                # 排序并显示组合结果
                if combo_results:
                    combo_results.sort(key=lambda x: x["评论数累计"], reverse=True)
                    combo_df = pd.DataFrame(combo_results)
                    st.dataframe(combo_df)
                    
                    # 可视化前10个组合
                    top_combos = min(10, len(combo_results))
                    if top_combos > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        chart_data = combo_df.head(top_combos)
                        sns.barplot(x="评论数累计", y="英文组合关键词", data=chart_data, ax=ax)
                        ax.set_title(f"Top {top_combos} 关键词组合评论数")
                        st.pyplot(fig)
                
                st.success("分析完成!")

st.markdown("""
---
### 使用指南

1. 上传包含产品数据的Excel文件
2. 输入核心关键词，例如"cat bed"、"yoga pants"或任何产品名称
3. 选择标题列和评论数量列
4. 点击"分析"按钮获取结果

分析结果将显示产品类型、功能特性、材质特征和适用对象四大类关键词的评论数统计，以及高评论量的关键词组合。
""")
