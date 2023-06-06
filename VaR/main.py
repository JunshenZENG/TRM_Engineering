from datetime import date
from data_processing.excel_input_data_processor import VaRExcelReader


def main():
    excel_object = VaRExcelReader(input_folder="input", file_name="var_input_data.xlsx")

    var_model = excel_object.get_var_model(date(2019, 11, 15))
    var = var_model.calculated_var()
    print(var)


if __name__ == '__main__':
    main()

