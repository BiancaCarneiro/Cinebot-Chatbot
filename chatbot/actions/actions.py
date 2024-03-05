# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#
#
class ActionHelloWorld(Action):
#
     def name(self) -> Text:
         return "action_hello_world"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         dispatcher.utter_message(text="Hello World!")

         return []

horarios = ["18:00", "21:00", "22:00"]

class ActionConsultarHorarios(Action):
    
    def name(self) -> Text:
        return "action_consultar_horarios"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        total_horas = "Horários disponíveis: "

        for horario in horarios:
            total_horas += horario + " | "  

        dispatcher.utter_message(text=total_horas)

        return []
    
class ActionVerificarHorario(Action):

    def name(self) -> Text:
        return "action_verificar_horarios"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        return []