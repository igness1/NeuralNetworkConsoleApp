# Wyjątek kiedy błąd sieci jest wystarczająco mały.
class ErrorWasEnoughSmall(Exception):
    def __init__(self):
        print("Error is enough small.")
class AccurancyDecreasedToManyTimes(Exception):
    def __init__(self):
        print("Accurancy values was decreasing to many times.")
class WrongSetNumberChoosen(Exception):
    def __init__(self):
        print("You have choosen wrong set number. Available options are values from 1 to 5.")
class ErrorHasTooLittleChanges(Exception):
    def __init__(self):
        print("The error stops decreasing.")