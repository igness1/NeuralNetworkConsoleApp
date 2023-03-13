from normalization import MinMaxScaler
import pandas as pd


def glass_database():
    target_names = ['building_windows_float_processed', 'building_windows_non_float_processed',
                    'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware',
                    'headlamps']
    column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "target"]
    glass = pd.read_csv('databases\glass.data', sep=',', names=column_names)
    glass = glass.drop("Id", axis=1)
    glass['target'].replace(to_replace={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}, inplace=True)
    glass_target_name = glass['target'].replace([0, 1, 2, 3, 4, 5, 6], [t for t in target_names])

    glass = glass.sample(frac=1).reset_index(drop=True)
    glass = MinMaxScaler(feature_range=(0, 1)).fit_transform(glass)


    glass_input = glass[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]].to_numpy()
    glass_output = glass['target'].to_numpy()

    return glass_input, glass_output, target_names


def breast_cancer_database():
    target_names = ['malignant', 'benign']
    column_names = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                    'mean fractal dimension',
                    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
                    'compactness error', 'concavity error', 'concave points error', 'symmetry error',
                    'fractal dimension error',
                    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
                    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
                    'worst fractal dimension']

    bcd = pd.read_csv('databases\wdbc.data', sep=',', names=column_names)
    bcd = bcd.drop("id", axis=1)
    bcd['diagnosis'].replace(to_replace={'M': 0, 'B': 1}, inplace=True)
    bcd_target_name = bcd['diagnosis'].replace([0, 1], [t for t in target_names])

    bcd = bcd.sample(frac=1).reset_index(drop=True)
    bcd = MinMaxScaler(feature_range=(0, 1)).fit_transform(bcd)

    bcd_input = bcd.drop('diagnosis', axis=1).to_numpy()
    bcd_output = bcd['diagnosis'].to_numpy()

    return bcd_input, bcd_output,target_names

def wine_database():
    target_names = ['class_0','class_1','class_2']
    column_names = ['wine_type', 'alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids',
                    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']
    wine_database = pd.read_csv('databases\wine.data',sep=',',names=column_names)
    wine_database_target_names = wine_database['wine_type'].replace([1,2,3],[t for t in target_names])

    wine_database = wine_database.sample(frac=1).reset_index(drop=True)
    wine_database = MinMaxScaler(feature_range=(0,1)).fit_transform(wine_database)

    wine_input = wine_database.drop('wine_type', axis=1).to_numpy()
    wine_output = wine_database['wine_type'].to_numpy()

    wine_output_target_name = MinMaxScaler(feature_range=(0,2)).fit_transform(wine_output)

    return wine_input,wine_output,target_names


