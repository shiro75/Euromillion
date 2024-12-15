import os
import pandas as pd

class ImportData:

    def __init__(self, pathfolder, filenames):
        self.pathfolder = pathfolder
        self.filenames = filenames
        self.dfs = []

    def importFile(self):
        if not os.path.exists(self.pathfolder):
            print(f"Directory does not exist: {self.pathfolder}")
        else:
            for filename in self.filenames:
                file_path = os.path.join(self.pathfolder, filename)

                # Check if the file exists before trying to read it
                if os.path.isfile(file_path):
                    self.dfs.append(pd.read_csv(file_path, delimiter=';'))
                else:
                    print(f"File does not exist: {file_path}")
        print("Data successfully imported")
        return self.dfs


    def mergeFile(self):
        if not self.dfs:
            print("No DataFrames to merge.")
            return None

            # Combine all DataFrames into one
        df_merge = pd.concat(self.dfs, ignore_index=True)

        print("Colonnes disponibles dans df_merge:", df_merge.columns.tolist())

        columns_to_display = [
            'annee_numero_de_tirage',
            'boule_1',
            'boule_2',
            'boule_3',
            'boule_4',
            'boule_5',
            'etoile_1',
            'etoile_2'
        ]

        missing_columns = [col for col in columns_to_display if col not in df_merge.columns]
        if missing_columns:
            print(f"Missing column in df_merge: {missing_columns}")
            return None

        df_final = df_merge[columns_to_display]
        df_final.iloc[::-1]
        print("DataFrame successfully created.")
        return df_final


