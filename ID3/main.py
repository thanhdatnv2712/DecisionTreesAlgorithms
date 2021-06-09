import numpy as np
import random
import pandas as pd
from pandas.io import parsers


#############################################ID3####################################################
#  TỔNG QUAN VỀ LUỒNG THỰC HIỆN CỦA CÂY QUYẾT ĐỊNH ID3                                             #
#  1. Duyệt for để lấy tất cả thuộc tính, giá trị duy nhất (unique_values) của bảng.               #
#  2. Với mỗi thuộc tính, kiểm tra xem là dạng liên tục hay rời rạc                                #
#  3. Với mỗi giá trị (value) của thuộc tính, tách bộ dữ liệu làm 2 phần:                          #
#      - Nếu thuộc tính liên tục: Phần 1 gồm các giá trị <= value; Phần 2 gồm các giá trị > value  #
#      - Nếu thuộc tính rời rạc: Phần 1 gồm các giá trị = value; Phần 2 gồm các giá trị != value   #
#  4. Tính overall_entropy sau khi tách                                                            #
#  5. Tách bộ dữ liệu làm 2 nhánh theo thuộc tính, giá trị làm cho overall_entropy nhỏ nhất        #
#  6. Lặp lại bước 1 đến khi thỏa mãn 1 trong 3 điều kiện:                                         #
#      - Trong nhánh chỉ chứa 1 nhãn                                                               #
#      - Độ sâu cây lớn hơn max_depth                                                              #
#      - Số dữ liệu trong nhánh nhỏ hơn min_sample                                                 #
####################################################################################################

FEATURE_TYPES = [] # categorical or continuous
COLUMN_HEADERS = None

#Kiểm tra dạng của thuộc tính là liên tục hay rời rạc
def check_type_of_attributes(df):
    feature_types = []
    n_unique_values_threshold = 10

    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")

    return feature_types

# Tách dữ liệu thành 2 bộ train và test theo tỉ lệ
def train_test_split(df, test_size):
    test_size = round(test_size * len(df))
    test_indices = random.sample(population=df.index.tolist(), k=test_size)
    validation_set = df.loc[test_indices]
    training_set = df.drop(test_indices)
    return training_set, validation_set

# Kiểm tra nút có tồn tại nhiều label hay không
def check_value(data):
    lanels = data[:, -1]
    unique_classes = np.unique(lanels)
    if len(unique_classes) == 1:
        return True
    else:
        return False

# Lấy ra nhãn có số lượng nhiều nhất trong data (Trong TH không thể phân thêm nhánh)
def get_most_common_label(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    LargestCountIndex = counts_unique_classes.argmax()
    classification = unique_classes[LargestCountIndex]
    return classification

# Lấy ra bộ thuộc tính hợp lệ để làm điều kiện tách nhánh
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape

    for column_index in range(n_columns - 1):
        values = data[:, column_index] 
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values
    return potential_splits

# Tách dữ liệu thành 2 phần theo thuộc tính
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above

# Hàm tính entropy
def entropy(data):
    labels = data[:, -1]
    _, label_counts = np.unique(labels, return_counts=True)

    p_x = label_counts / label_counts.sum()
    entropy = sum(p_x * - np.log2(p_x))
    return entropy

# Hàm tính entropy của tất cả các nhánh sau khi phân chia
def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_entropy = (p_data_below * entropy(data_below) + p_data_above * entropy(data_above))

    return overall_entropy

# Hàm tìm thuộc tính và gía trị để phân nhánh sao cho sau khi phân chia, overall_entropy đạt nhỏ nhất
def determine_best_split(data, potential_splits):
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

# Cây ID3 
def Tree_ID3(df, counter=0, min_samples=2, max_depth=5):
    global COLUMN_HEADERS, FEATURE_TYPES
    if counter == 0:
        data = df.values
    else:
        data = df

    if (check_value(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = get_most_common_label(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)

        best_split_column, best_split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, best_split_column, best_split_value)

        if len(data_below) == 0 or len(data_above) == 0:
            classification = get_most_common_label(data)
            return classification

        feature_name = COLUMN_HEADERS[best_split_column]
        type_of_feature = FEATURE_TYPES[best_split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, best_split_value)
        else:
            question = "{} = {}".format(feature_name, best_split_value)

        node = {question: []}

        true_answer = Tree_ID3(data_below, counter, min_samples, max_depth)
        false_answer = Tree_ID3(data_above, counter, min_samples, max_depth)

        if true_answer == false_answer:
            node = true_answer
        else:
            node[question].append(false_answer)
            node[question].append(true_answer)

        return node

# Hàm dự đoán nhãn
def predict(sample, tree):
    random.seed(1234)
    question = list(tree.keys())[0]
    feature_name, operator, value = question.split()

    if operator == "<=":
        if sample[feature_name] <= float(value):
            answer = tree[question][1]
        else:
            answer = tree[question][0]
    else:
        if str(sample[feature_name]) == value:
            answer = tree[question][1]
        else:
            answer = tree[question][0]

    if isinstance(answer, dict):
        return predict(sample, answer)
    else:
        return answer

# Hàm tính độ chính xác        
def calc_accuracy(df, tree):
    count = 0
    for row in df.iloc:
        if predict(row, tree) == row[-1]:
            count+=1
    return count/len(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="dataset/weather.csv")
    parser.add_argument("--min_sample", "-ms", type=int, default=2)
    parser.add_argument("--max_depth", "-md", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    COLUMN_HEADERS = df.columns
    FEATURE_TYPES = check_type_of_attributes(df)    
    df = df.rename(columns={COLUMN_HEADERS[-1]: "label"})
    training_set, validation_set = train_test_split(df, 0.2)
    tree = Tree_ID3(training_set, 0, args.min_sample, args.max_depth)
    print(tree)
    print(calc_accuracy(validation_set, tree))
