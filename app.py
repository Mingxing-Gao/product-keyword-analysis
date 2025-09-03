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
        "​
Add app code
