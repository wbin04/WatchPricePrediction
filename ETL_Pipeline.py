import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

class ETLPipeline:
    def __init__(self, input_file='datasets/watchbase_data_raw.csv', output_file='datasets/watchbase_data_etl_transformed.csv'):
        self.data = None
        self.input_file = input_file
        self.output_file = output_file

    def data_preprocessing(self):
        # Simulate data preprocessing
        print("Preprocessing data...")

        self.data.columns = self.data.columns.str.replace('_', ' ')

        self.data = self.data.drop(columns=['Produced'])

        self.data['Case Material'] = self.data['Case Material'].apply(lambda x: 'Stainless Steel' if pd.isna(x) else x)

        self.data = self.data.drop(columns=['Lug Width'])

        self.data['Water Resistance'] = self.data['Water Resistance'].str.replace(' m', '').astype(float)
        self.data['Water Resistance'] = self.data['Water Resistance'].fillna(self.data['Water Resistance'].mode()[0])

        self.data = self.data.drop(columns=['Dial Finish'])

        self.data['Dial Indexes'] = self.data['Dial Indexes'].fillna(self.data['Dial Indexes'].mode()[0])

        self.data['Dial Hands'] = self.data['Dial Hands'].fillna(self.data['Dial Hands'].mode()[0])

        self.data['Price'] = self.data['Price'].astype(float)

        self.data = self.data.rename(columns={'Family': 'Model'})
        self.data = self.data.drop(columns=['Reference', 'Name'])

        self.data['Limited'] = self.data['Limited'].str.split(',').str[0]

        self.data['Glass'] = self.data['Glass'].fillna(self.data['Glass'].mode()[0])

        self.data['Case Back'] = self.data['Case Back'].fillna(self.data['Case Back'].mode()[0])

        self.data['Case Shape'] = self.data['Case Shape'].fillna(self.data['Case Shape'].mode()[0])

        self.data['Case Diameter'] = self.data['Case Diameter'].str.replace(' mm', '').astype(float)
        self.data['Case Diameter'] = self.data['Case Diameter'].fillna(self.data['Case Diameter'].mode()[0])

        self.data['Dial Color'] = self.data['Dial Color'].fillna(self.data['Dial Color'].mode()[0])

        return self.data
    
    def feature_engineering(self):
        # Simulate feature engineering
        print("Performing feature engineering...")

        self.data = self.data[self.data['Case Diameter'] <= 60]

        bins = [0, 30, 100, 200, 500, float('inf')]
        labels = ['Low', 'Basic', 'Standard', 'Professional', 'Extreme']
        self.data['WaterResistanceLevel'] = pd.cut(self.data['Water Resistance'], bins=bins, labels=labels, right=False)

        self.data['Brand'] = self.data['Brand'].str.replace('-', ' ').str.lower()
        self.data['Brand'] = self.data['Brand'].str.replace('bell ross', 'bell & ross')

        self.data['Model'] = self.data['Model'].str.replace(' watches', '')

        def remove_brand_from_model(model_name):
            brands = self.data['Brand'].unique().tolist()
            for brand in brands:
                model_name = re.sub(rf"^{re.escape(brand)}\s*", "", model_name)
            return model_name

        self.data['Model'] = self.data['Model'].apply(lambda x: remove_brand_from_model(x.lower().strip()))

        def group_case_material(material):
            if material in ['Stainless Steel']:
                return 'Steel'
            elif material in ['Titanium']:
                return 'Titanium'
            elif material in ['White Gold', 'Pink Gold', 'Rose Gold', 'Red Gold', 'Yellow Gold', 'Goldtech', 'Sedna Gold', 'Bronze Gold']:
                return 'Gold Variants'
            elif material == 'Platinum':
                return 'Platinum'
            elif material == 'Bronze':
                return 'Bronze'
            elif material in ['Ceramic', 'Carbon', 'Sapphire', 'Resin', 'Mother of Pearl']:
                return 'Synthetic'
            elif material == 'Diamond':
                return 'Diamond'
            else:
                return 'Other'

        self.data['CaseMaterialGrouped'] = self.data['Case Material'].apply(group_case_material)

        self.data['LogPrice'] = np.log1p(self.data['Price'])

        def group_dial_color(color):
            color = color.lower()
            if color in ['black']:
                return 'Black'
            elif color in ['silver', 'white', 'champagne', 'ivory', 'mirror']:
                return 'Silver/White'
            elif color in ['blue', 'navy']:
                return 'Blue'
            elif color in ['skeleton', 'see-through']:
                return 'Skeleton'
            elif color in ['grey', 'brown', 'taupe', 'beige']:
                return 'Grey/Brown'
            elif color in ['paved', 'diamonds']:
                return 'Paved/Diamonds'
            elif color in ['multi-color', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'salmon', 'rose']:
                return 'Colorful'
            else:
                return 'Other'

        self.data['DialColorGrouped'] = self.data['Dial Color'].apply(group_dial_color)

        def group_dial_hands(hand):
            hand = hand.lower()
            if hand in ['stick', 'baton', 'alpha']:
                return 'Stick/Minimalist'
            elif hand in ['dauphine', 'lancette', 'trapezium']:
                return 'Dauphine-style'
            elif hand in ['sword', 'arrow']:
                return 'Sword-style'
            elif hand in ['feuille', 'poire']:
                return 'Leaf-style'
            elif hand in ['cathedrale', 'mercedes', 'breguet', 'syringe', 'losange']:
                return 'Traditional/Special'
            elif hand in ['proprietary']:
                return 'Proprietary'
            else:
                return 'Other'

        self.data['DialHandsGrouped'] = self.data['Dial Hands'].apply(group_dial_hands)

        bins = [0, 30, 34, 39, 43, 47, float('inf')]
        labels = ['XS', 'S', 'M', 'L', 'XL', 'XXL']

        self.data['CaseDiameterGrouped'] = pd.cut(self.data['Case Diameter'], bins=bins, labels=labels)

        self.data = self.data[self.data['LogPrice'].notna()]

        return self.data

    def extract(self):
        # Simulate data extraction
        print("Extracting data...")
        self.data = pd.read_csv(self.input_file)
        self.data.to_parquet("data_lake/watch_raw.parquet", index=False)
        return self.data

    def transform(self):
        # Simulate data transformation
        print("Transforming data...")
        self.data_preprocessing()
        self.feature_engineering()
        return self.data

    def load(self):
        # Simulate loading data to a destination
        print("Loading data...")
        cols = ['Url', 'Brand', 'Model', 'Limited', 
            'CaseMaterialGrouped', 'Glass', 'Case Shape', 'CaseDiameterGrouped', 'WaterResistanceLevel', 
            'DialColorGrouped', 'DialHandsGrouped', 'Dial Indexes', 'LogPrice']
        self.data = self.data[cols]
        print(self.data.head())

        self.data.to_csv(self.output_file, index=False)

        engine = create_engine('sqlite:///data_warehouse/watch_dwh.db')
        self.data.to_sql('watch_info', engine, index=False, if_exists='replace')

    def run_etl_pipeline(self):
        print("Running ETL Pipeline...")
        self.extract()
        self.transform()
        self.load()
        print("ETL Pipeline completed successfully.")

if __name__ == "__main__":
    input_file_path = 'datasets_etl/watchbase_data_raw.csv'
    output_file_path = 'datasets_etl/watchbase_data_transformed.csv'

    etl_pipeline = ETLPipeline(input_file=input_file_path, output_file=output_file_path)
    etl_pipeline.run_etl_pipeline()