from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM clean_data', engine)
print(df.head())