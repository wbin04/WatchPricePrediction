import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re

class SparkETLPipeline:
    def __init__(self, 
                 input_file='datasets_etl/data_raw.csv', 
                 output_file='datasets_etl/data_transformed_spark.csv', 
                 output_dl='data_lake/watch_dl_spark.parquet', 
                 output_dwh='data_warehouse/watch_dwh_spark.db'):
        self.data = None
        self.input_file = input_file
        self.output_file = output_file
        self.output_dl = output_dl
        self.output_dwh = output_dwh
        
        # Initialize Spark
        try:
            self.spark.stop()
        except:
            pass
        
        self.spark = SparkSession.builder \
            .appName("Watch Price Prediction ETL") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

    def data_preprocessing(self):
        """Preprocess data using Spark DataFrame operations"""
        print("Preprocessing data...")
        
        # Replace underscores with spaces in column names
        for col_name in self.data.columns:
            if '_' in col_name:
                self.data = self.data.withColumnRenamed(col_name, col_name.replace('_', ' '))
        
        # Drop 'Produced' column
        self.data = self.data.drop('Produced')
        
        # Fill null values in 'Case Material' with 'Stainless Steel'
        self.data = self.data.fillna({'Case Material': 'Stainless Steel'})
        
        # Drop 'Lug Width' column
        self.data = self.data.drop('Lug Width')
        
        # Clean 'Water Resistance' column
        self.data = self.data.withColumn('Water Resistance', 
                                        regexp_replace('Water Resistance', ' m', '').cast(DoubleType()))
        
        # Fill null values in 'Water Resistance' with mode
        water_resistance_mode = self.data.groupBy('Water Resistance').count().orderBy(desc('count')).first()['Water Resistance']
        self.data = self.data.fillna({'Water Resistance': water_resistance_mode})
        
        # Drop 'Dial Finish' column
        self.data = self.data.drop('Dial Finish')
        
        # Fill null values in 'Dial Indexes' with mode
        dial_indexes_mode = self.data.groupBy('Dial Indexes').count().orderBy(desc('count')).first()['Dial Indexes']
        self.data = self.data.fillna({'Dial Indexes': dial_indexes_mode})
        
        # Fill null values in 'Dial Hands' with mode
        dial_hands_mode = self.data.groupBy('Dial Hands').count().orderBy(desc('count')).first()['Dial Hands']
        self.data = self.data.fillna({'Dial Hands': dial_hands_mode})
        
        # Convert 'Price' to float
        self.data = self.data.withColumn('Price', col('Price').cast(DoubleType()))
        
        # Rename 'Family' to 'Model'
        self.data = self.data.withColumnRenamed('Family', 'Model')
        
        # Drop 'Reference' and 'Name' columns
        self.data = self.data.drop('Reference', 'Name')
        
        # Clean 'Limited' column
        self.data = self.data.withColumn('Limited', split(col('Limited'), ',').getItem(0))
        
        # Fill null values in 'Glass' with mode
        glass_mode = self.data.groupBy('Glass').count().orderBy(desc('count')).first()['Glass']
        self.data = self.data.fillna({'Glass': glass_mode})
        
        # Fill null values in 'Case Back' with mode
        case_back_mode = self.data.groupBy('Case Back').count().orderBy(desc('count')).first()['Case Back']
        self.data = self.data.fillna({'Case Back': case_back_mode})
        
        # Fill null values in 'Case Shape' with mode
        case_shape_mode = self.data.groupBy('Case Shape').count().orderBy(desc('count')).first()['Case Shape']
        self.data = self.data.fillna({'Case Shape': case_shape_mode})
        
        # Clean 'Case Diameter' column
        self.data = self.data.withColumn('Case Diameter', 
                                        regexp_replace('Case Diameter', ' mm', '').cast(DoubleType()))
        
        # Fill null values in 'Case Diameter' with mode
        case_diameter_mode = self.data.groupBy('Case Diameter').count().orderBy(desc('count')).first()['Case Diameter']
        self.data = self.data.fillna({'Case Diameter': case_diameter_mode})
        
        # Fill null values in 'Dial Color' with mode
        dial_color_mode = self.data.groupBy('Dial Color').count().orderBy(desc('count')).first()['Dial Color']
        self.data = self.data.fillna({'Dial Color': dial_color_mode})
        
        return self.data
    
    def feature_engineering(self):
        """Perform feature engineering using Spark DataFrame operations"""
        print("Performing feature engineering...")
        
        # Filter out watches with Case Diameter > 60
        self.data = self.data.filter(col('Case Diameter') <= 60)
        
        # Create Water Resistance Level categories
        self.data = self.data.withColumn('WaterResistanceLevel',
                                        when(col('Water Resistance') < 30, 'Low')
                                        .when(col('Water Resistance') < 100, 'Basic')
                                        .when(col('Water Resistance') < 200, 'Standard')
                                        .when(col('Water Resistance') < 500, 'Professional')
                                        .otherwise('Extreme'))
        
        # Clean Brand column
        self.data = self.data.withColumn('Brand', 
                                        regexp_replace(lower(col('Brand')), '-', ' '))
        self.data = self.data.withColumn('Brand', 
                                        regexp_replace(col('Brand'), 'bell ross', 'bell & ross'))
        
        # Clean Model column
        self.data = self.data.withColumn('Model', 
                                        regexp_replace(col('Model'), ' watches', ''))
        
        # Create CaseMaterialGrouped column
        self.data = self.data.withColumn('CaseMaterialGrouped',
                                        when(col('Case Material') == 'Stainless Steel', 'Steel')
                                        .when(col('Case Material') == 'Titanium', 'Titanium')
                                        .when(col('Case Material').isin(['White Gold', 'Pink Gold', 'Rose Gold', 'Red Gold', 'Yellow Gold', 'Goldtech', 'Sedna Gold', 'Bronze Gold']), 'Gold Variants')
                                        .when(col('Case Material') == 'Platinum', 'Platinum')
                                        .when(col('Case Material') == 'Bronze', 'Bronze')
                                        .when(col('Case Material').isin(['Ceramic', 'Carbon', 'Sapphire', 'Resin', 'Mother of Pearl']), 'Synthetic')
                                        .when(col('Case Material') == 'Diamond', 'Diamond')
                                        .otherwise('Other'))
        
        # Create LogPrice column
        self.data = self.data.withColumn('LogPrice', log1p(col('Price')))
        
        # Create DialColorGrouped column
        self.data = self.data.withColumn('DialColorGrouped',
                                        when(lower(col('Dial Color')) == 'black', 'Black')
                                        .when(lower(col('Dial Color')).isin(['silver', 'white', 'champagne', 'ivory', 'mirror']), 'Silver/White')
                                        .when(lower(col('Dial Color')).isin(['blue', 'navy']), 'Blue')
                                        .when(lower(col('Dial Color')).isin(['skeleton', 'see-through']), 'Skeleton')
                                        .when(lower(col('Dial Color')).isin(['grey', 'brown', 'taupe', 'beige']), 'Grey/Brown')
                                        .when(lower(col('Dial Color')).isin(['paved', 'diamonds']), 'Paved/Diamonds')
                                        .when(lower(col('Dial Color')).isin(['multi-color', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'salmon', 'rose']), 'Colorful')
                                        .otherwise('Other'))
        
        # Create DialHandsGrouped column
        self.data = self.data.withColumn('DialHandsGrouped',
                                        when(lower(col('Dial Hands')).isin(['stick', 'baton', 'alpha']), 'Stick/Minimalist')
                                        .when(lower(col('Dial Hands')).isin(['dauphine', 'lancette', 'trapezium']), 'Dauphine-style')
                                        .when(lower(col('Dial Hands')).isin(['sword', 'arrow']), 'Sword-style')
                                        .when(lower(col('Dial Hands')).isin(['feuille', 'poire']), 'Leaf-style')
                                        .when(lower(col('Dial Hands')).isin(['cathedrale', 'mercedes', 'breguet', 'syringe', 'losange']), 'Traditional/Special')
                                        .when(lower(col('Dial Hands')) == 'proprietary', 'Proprietary')
                                        .otherwise('Other'))
        
        # Create CaseDiameterGrouped column
        self.data = self.data.withColumn('CaseDiameterGrouped',
                                        when(col('Case Diameter') <= 30, 'XS')
                                        .when(col('Case Diameter') <= 34, 'S')
                                        .when(col('Case Diameter') <= 39, 'M')
                                        .when(col('Case Diameter') <= 43, 'L')
                                        .when(col('Case Diameter') <= 47, 'XL')
                                        .otherwise('XXL'))
        
        # Filter out null LogPrice values
        self.data = self.data.filter(col('LogPrice').isNotNull())
        
        return self.data

    def extract(self):
        """Extract data from CSV file"""
        print("Extracting data...")
        self.data = self.spark.read.csv(self.input_file, header=True, inferSchema=True)
        
        # Save to data lake as parquet
        self.data.write.mode('overwrite').parquet(self.output_dl)
        
        return self.data

    def transform(self):
        """Transform data through preprocessing and feature engineering"""
        print("Transforming data...")
        self.data_preprocessing()
        self.feature_engineering()
        return self.data

    def load(self):
        """Load data to destination files"""
        print("Loading data...")
        
        # Select final columns
        final_columns = ['Url', 'Brand', 'Model', 'Limited', 
                        'CaseMaterialGrouped', 'Glass', 'Case Shape', 'CaseDiameterGrouped', 
                        'WaterResistanceLevel', 'DialColorGrouped', 'DialHandsGrouped', 
                        'Dial Indexes', 'LogPrice']
        
        self.data = self.data.select(final_columns)
        
        # Show sample data
        print("Sample transformed data:")
        self.data.show(5)
        
        # Save to CSV
        self.data.coalesce(1).write.mode('overwrite').option('header', 'true').csv(self.output_file)
        
        # Save to SQLite database (convert to pandas for SQLite compatibility)
        pandas_df = self.data.toPandas()
        from sqlalchemy import create_engine
        engine = create_engine(f'sqlite:///{self.output_dwh}')
        pandas_df.to_sql('watch_info', engine, index=False, if_exists='replace')

    def run_etl_pipeline(self):
        """Run the complete ETL pipeline"""
        print("Running Spark ETL Pipeline...")
        self.extract()
        self.transform()
        self.load()
        print("Spark ETL Pipeline completed successfully.")
        
    def stop_spark(self):
        """Stop Spark session"""
        self.spark.stop()

    def read_from_data_lake(self):
        """Read data from data lake (parquet format)"""
        print(f"Reading data from data lake: {self.output_dl}")
        try:
            df = self.spark.read.parquet(self.output_dl)
            print(f"Successfully loaded {df.count()} rows from data lake")
            print("Schema:")
            df.printSchema()
            print("\nSample data:")
            df.show(5)
            return df
        except Exception as e:
            print(f"Error reading from data lake: {e}")
            return None

    def read_from_transformed_csv(self):
        """Read data from transformed CSV files"""
        print(f"Reading data from transformed CSV: {self.output_file}")
        try:
            df = self.spark.read.csv(self.output_file, header=True, inferSchema=True)
            print(f"Successfully loaded {df.count()} rows from CSV")
            print("Schema:")
            df.printSchema()
            print("\nSample data:")
            df.show(5)
            return df
        except Exception as e:
            print(f"Error reading from CSV: {e}")
            return None

if __name__ == "__main__":
    input_file_path = 'datasets_etl/data_raw.csv'
    output_file_path = 'datasets_etl/data_transformed_spark.csv'
    output_dl_path = 'data_lake/watch_dl_spark.parquet'
    output_dwh_path = 'data_warehouse/watch_dwh_spark.db'

    etl_pipeline = SparkETLPipeline(input_file=input_file_path, 
                                   output_file=output_file_path, 
                                   output_dl=output_dl_path, 
                                   output_dwh=output_dwh_path)
    
    try:
        etl_pipeline.run_etl_pipeline()
            
    finally:
        etl_pipeline.stop_spark()