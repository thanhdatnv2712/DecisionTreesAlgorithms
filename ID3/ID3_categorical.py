import numpy as np
import random
import pandas as pd

#############################################ID3####################################################
#  TỔNG QUAN VỀ LUỒNG THỰC HIỆN CỦA CÂY QUYẾT ĐỊNH ID3                                             #
#  1. Duyệt for để lấy tất cả thuộc tính, giá trị duy nhất (unique_values) của thuộc tính đó.      #
#  2. Với mỗi  thuộc tính, tách bộ dữ liệu làm n phần với n = unique_values:                       #
#  3. Tính overall_entropy sau khi tách                                                            #
#  4. Tách bộ dữ liệu làm n nhánh theo thuộc tính làm cho overall_entropy nhỏ nhất                 #
#  5. Lặp lại bước 1 đến khi thỏa mãn 1 trong 3 điều kiện:                                         #
#      - Trong nhánh chỉ chứa 1 nhãn                                                               #
#      - Độ sâu cây lớn hơn max_depth                                                              #
#      - Số dữ liệu trong nhánh nhỏ hơn min_sample                                                 #
####################################################################################################

COLUMN_HEADERS = None

# Tách dữ liệu thành 2 bộ train và test theo tỉ lệ
def train_test_split(data, test_size):
    test_size = round(test_size * len(data))
    test_indices = random.sample(population=data.index.tolist(), k=test_size)
    validation_set = data.loc[test_indices]
    training_set = data.drop(test_indices)
    return training_set, validation_set

# Kiểm tra nút có tồn tại nhiều label hay không
def check_value(data):
    labels = data[:, -1]
    unique_classes = np.unique(labels)
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
def split_data(data, column_index):
    split_column_values = data[:, column_index]
    values = data[:, column_index] 
    unique_values = np.unique(values)
    splitted_data = []
    for value in unique_values:
        data_below = data[split_column_values == value]
        splitted_data.append(data_below)
    return splitted_data

# Hàm tính entropy
def entropy(data):
    labels = data[:, -1]
    _, label_counts = np.unique(labels, return_counts=True)

    p_x = label_counts / label_counts.sum()
    entropy = sum(p_x * - np.log2(p_x))
    return entropy

# Hàm tính entropy của tất cả các nhánh sau khi phân chia
def calculate_overall_entropy(splitted_data):
    n_data_points = 0
    for data in splitted_data:
        n_data_points += len(data)
    overall_entropy = 0    
    for data in splitted_data:
        p_x = len(data) / n_data_points
        overall_entropy += p_x*entropy(data)
    return overall_entropy

# Hàm tìm thuộc tính và gía trị để phân nhánh sao cho sau khi phân chia, overall_entropy đạt nhỏ nhất
def determine_best_split(data, potential_splits):
    overall_entropy = 999
    for column_index in potential_splits:
        splitted_data = split_data(data, column_index)
        current_overall_entropy = calculate_overall_entropy(splitted_data)

        if current_overall_entropy <= overall_entropy:
            overall_entropy = current_overall_entropy
            best_split_column = column_index

    return best_split_column

# Cây ID3 
def Tree_ID3(data, counter=0, min_samples=2, max_depth=5):
    global COLUMN_HEADERS, FEATURE_TYPES
    if counter == 0:
        data = data.values
    else:
        data = data

    if (check_value(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = get_most_common_label(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        best_split_column = determine_best_split(data, potential_splits)
        splitted_data = split_data(data, best_split_column)

        for temp_data in splitted_data:
            if len(temp_data) == 0:
                classification = get_most_common_label(data)
                return classification

        feature_name = COLUMN_HEADERS[best_split_column]
        root_question = "{} ?".format(feature_name)
        root_node = []
        for temp_data in splitted_data:
            value = temp_data[0][best_split_column]
            question = "{} = {}".format(feature_name, value)
            node = {
                question: []
            }
            leaf = Tree_ID3(temp_data, counter, min_samples, max_depth)
            node[question].append(leaf)
            root_node.append(node)
        return root_node

# Hàm dự đoán nhãn
def predict(sample, tree):
    random.seed(1234)
    for att in tree:
        question = list(att.keys())[0]
        feature_name, _, value = question.split()
        if str(sample[feature_name]) == value:
            answer = att[question][0]
    if isinstance(answer, list):
        return predict(sample, answer)
    else:
        return answer

# Hàm tính độ chính xác        
def calc_accuracy(data, tree):
    count = 0
    for row in data.iloc:
        if predict(row, tree) == row[-1]:
            count+=1
    return count/len(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="../dataset/weather.csv")
    parser.add_argument("--min_sample", "-ms", type=int, default=2)
    parser.add_argument("--max_depth", "-md", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    COLUMN_HEADERS = data.columns    
    data = data.rename(columns={COLUMN_HEADERS[-1]: "label"})
    training_set, validation_set = train_test_split(data, 0.2)
    tree = Tree_ID3(training_set, 0, args.min_sample, args.max_depth)
    print(tree)
    print(calc_accuracy(validation_set, tree))
