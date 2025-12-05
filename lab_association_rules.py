import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

# 1. Загрузка данных с разными вариантами разделителя
try:
    # сначала пробуем стандартный вариант Kaggle
    all_data = pd.read_csv('groceries.csv')
except pd.errors.ParserError:
    # если не получилось – скорее всего, файл с разделителем ';'
    all_data = pd.read_csv('groceries.csv', sep=';', engine='python')

print("Первые строки исходного датасета:")
print(all_data.head())
print("\nКолонки датасета:")
print(all_data.columns)
print("\nИнформация о датасете:")
print(all_data.info())


# Ожидаем, что есть колонка с названием товара
# В большинстве версий датасета это 'itemDescription'
col_name = all_data.columns[0]

# Преобразуем каждую строку в список товаров
transactions = (
    all_data[col_name]
    .dropna()
    .apply(lambda row: [item.strip() for item in str(row).split(',')])
    .tolist()
)

# 3. Анализ длин транзакций
transaction_lengths = [len(t) for t in transactions]

print(f"\nВсего транзакций: {len(transactions)}")
print(f"Мин. длина транзакции: {min(transaction_lengths)}")
print(f"Макс. длина транзакции: {max(transaction_lengths)}")
print(f"Средняя длина транзакции: {sum(transaction_lengths) / len(transaction_lengths):.2f}")

plt.figure(figsize=(8, 4))
plt.hist(transaction_lengths, bins=range(1, max(transaction_lengths) + 2))
plt.xlabel('Длина транзакции (количество товаров)')
plt.ylabel('Частота')
plt.title('Распределение длин транзакций')
plt.tight_layout()
plt.show()

# 4. Список уникальных товаров
from itertools import chain

all_items = list(chain.from_iterable(transactions))
unique_items = sorted(set(all_items))

print(f"\nКоличество уникальных товаров: {len(unique_items)}")
print("Первые 20 товаров:")
print(unique_items[:20])

# 5. One-hot encoding (TransactionEncoder)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
data = pd.DataFrame(te_ary, columns=te.columns_)

print("\nПервые строки бинарной матрицы транзакций:")
print(data.head())
print("\nРазмер матрицы data:", data.shape)
min_support = 0.02

frequent_itemsets_fpg = fpgrowth(
    data,
    min_support=min_support,
    use_colnames=True
)

# Добавим колонку "length" – размер набора
frequent_itemsets_fpg['length'] = frequent_itemsets_fpg['itemsets'].apply(len)

print("\n=== Частые наборы (FP-Growth) ===")
print(frequent_itemsets_fpg.head(20))
print("\nВсего частых наборов при min_support =", min_support, ":", len(frequent_itemsets_fpg))

# ==========================================================
# 7. Генерация ассоциативных правил из частых наборов
# ==========================================================

# min_threshold – это минимальное значение метрики (по умолчанию confidence)
min_confidence = 0.3

rules_fpg = association_rules(
    frequent_itemsets_fpg,
    metric="confidence",
    min_threshold=min_confidence
)

# Отсортируем правила по лифту (чтобы видеть наиболее "нетривиальные")
rules_fpg = rules_fpg.sort_values(by='lift', ascending=False)

print("\n=== Ассоциативные правила (FP-Growth, отсортированы по lift) ===")
print(rules_fpg.head(20))

print("\nВсего правил при min_support =", min_support,
      "и min_confidence =", min_confidence, ":", len(rules_fpg))

# Для удобства выведем несколько правил в более читаемом виде
def format_rule(row):
    return f"{set(row['antecedents'])} -> {set(row['consequents'])} | support={row['support']:.3f}, confidence={row['confidence']:.3f}, lift={row['lift']:.3f}"

print("\nПримеры правил:")
for _, row in rules_fpg.head(10).iterrows():
    print(format_rule(row))

# ==========================================================
# 8. Дополнительная визуализация правил: scatter support-confidence
# ==========================================================

plt.figure(figsize=(6, 5))
plt.scatter(rules_fpg['support'], rules_fpg['confidence'], alpha=0.6)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Правила (FP-Growth): Support vs Confidence')
plt.grid(True)
plt.tight_layout()
plt.show()

min_support_apr = 0.02   # тот же порог, что и для FP-Growth, для сравнения
frequent_itemsets_apr = apriori(
    data,
    min_support=min_support_apr,
    use_colnames=True
)
frequent_itemsets_apr['length'] = frequent_itemsets_apr['itemsets'].apply(len)

print("\n=== Частые наборы (Apriori) ===")
print(frequent_itemsets_apr.head(20))
print("Всего частых наборов (Apriori) при min_support =",
      min_support_apr, ":", len(frequent_itemsets_apr))

min_conf_apr = 0.3
rules_apr = association_rules(
    frequent_itemsets_apr,
    metric="confidence",
    min_threshold=min_conf_apr
)
rules_apr = rules_apr.sort_values(by='lift', ascending=False)

print("\n=== Ассоциативные правила (Apriori, отсортированы по lift) ===")
print(rules_apr.head(20))
print("Всего правил (Apriori) при min_support =",
      min_support_apr, "и min_confidence =", min_conf_apr, ":", len(rules_apr))

print("\nПримеры правил (Apriori):")
for _, row in rules_apr.head(10).iterrows():
    print(f"{set(row['antecedents'])} -> {set(row['consequents'])} | "
          f"support={row['support']:.3f}, confidence={row['confidence']:.3f}, lift={row['lift']:.3f}")
    

# 10. Минимальные поддержки для наборов разной длины (Apriori)
# ==========================================================

# Берем маленький порог, чтобы собрать много наборов
frequent_itemsets_apr_full = apriori(
    data,
    min_support=0.005,   # 0.5% транзакций
    use_colnames=True
)
frequent_itemsets_apr_full['length'] = frequent_itemsets_apr_full['itemsets'].apply(len)

support_stats_by_len = (
    frequent_itemsets_apr_full
    .groupby('length')['support']
    .agg(['count', 'min', 'max', 'mean'])
    .reset_index()
)

print("\nСтатистика поддержки по длине набора (Apriori, min_support=0.005):")
print(support_stats_by_len)

# 11. Эксперименты с параметрами min_support и min_confidence (FP-Growth)
# ==========================================================

support_values = [0.01, 0.02, 0.03, 0.05]
confidence_values = [0.3, 0.4, 0.5]

results = []

for sup in support_values:
    fi = fpgrowth(data, min_support=sup, use_colnames=True)
    for conf in confidence_values:
        r = association_rules(fi, metric="confidence", min_threshold=conf)
        results.append({
            'min_support': sup,
            'min_confidence': conf,
            'n_itemsets': len(fi),
            'n_rules': len(r),
            'mean_lift': r['lift'].mean() if len(r) > 0 else None
        })

results_df = pd.DataFrame(results)
print("\nРезультаты экспериментов (FP-Growth):")
print(results_df)

# Визуализация зависимости количества правил от порога поддержки
plt.figure(figsize=(6, 4))
for conf in confidence_values:
    subset = results_df[results_df['min_confidence'] == conf]
    plt.plot(subset['min_support'], subset['n_rules'],
             marker='o', label=f'conf >= {conf}')
plt.xlabel('min_support')
plt.ylabel('Количество правил')
plt.title('Зависимость количества правил от min_support (FP-Growth)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Граф ассоциативных правил (по FP-Growth)
# ==========================================================

# Возьмем, например, топ-30 правил по лифту
top_rules = rules_fpg.head(30)

G = nx.DiGraph()

for _, row in top_rules.iterrows():
    antecedents = tuple(row['antecedents'])
    consequents = tuple(row['consequents'])

    ant = ', '.join(antecedents)
    cons = ', '.join(consequents)

    G.add_node(ant)
    G.add_node(cons)
    G.add_edge(ant, cons,
               weight=row['confidence'],
               lift=row['lift'])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5, seed=42)

edges = G.edges(data=True)
edge_widths = [d['weight'] * 2 for (_, _, d) in edges]

nx.draw_networkx_nodes(G, pos, node_size=800)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, arrowstyle='->')

plt.title('Граф ассоциативных правил (FP-Growth, top-30 по lift)')
plt.axis('off')
plt.tight_layout()
plt.show()