# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import sys
sys.path.insert(0, '../')
from scraper.cinema_scraper import CinemaSpider

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

spider = CinemaSpider()

class ActionConfirmacaoFinal(Action):

     def name(self) -> Text:
         return "action_ask_confirmacaoFinal"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
         
        user_filme = tracker.get_slot("nomeFilme")
        user_time = tracker.get_slot("horarioFilme")

        dispatcher.utter_message(text=f"Marcar o filme {user_filme} no horário {user_time} então?")

        return []


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


class ActionVerificarFilme(Action):

    def name(self) -> Text:
        return "action_verificar_filme"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies = spider.get_times_dir()
        user_filme = next(tracker.get_latest_entity_values("filme"), None)
        
        format_movies = ""
        for i, mv in enumerate(list(movies.keys())):
            format_movies += f"{mv:^40}"
            if i % 2 == 1:
                format_movies += "\n"
            else:
                format_movies += " | "
        
        
        if not user_filme or user_filme not in movies.keys():
            dispatcher.utter_message(text=f"Digite um filme válido, os filmes em cartaz são:\n{format_movies}")
        else:
            dispatcher.utter_message(text=f"Certo, você quer ver o filme {user_filme}, qual horário você quer? Os horários disponíveis são: {[' ,'.join(movies[user_filme])]}")
        return []
    
class ActionVerificarHorario(Action):

    def name(self) -> Text:
        return "action_verificar_horarios"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies = spider.get_times_dir()
        user_filme = next(tracker.get_latest_entity_values("filme"), None)
        user_time = next(tracker.get_latest_entity_values("horario"), None)
        
        if not user_time or user_time not in movies.keys() or user_time not in movies[user_filme]:
            dispatcher.utter_message(text=f"Digite um horário válido, os horários são: {[' ,'.join(movies[user_filme])]}")
        else:
            dispatcher.utter_message(text=f"Certo, você quer ver o filme {user_time}, no horário {user_time}, combinado!")
        return []
    
class ActionPerguntaFilmes(Action):

    def name(self) -> Text:
        return "action_ask_filmes"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies = spider.get_times_dir()
        
        format_movies = ""
        for i, mv in enumerate(list(movies.keys())):
            format_movies += f"{mv:^40}"
            if i % 2 == 1:
                format_movies += "\n"
            else:
                format_movies += " | "
        
        dispatcher.utter_message(text=f"Os filmes em cartaz são:\n{format_movies}")
        
        return []
    
class ActionPerguntaPrecos(Action):

    def name(self) -> Text:
        return "action_ask_precos"
    
    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        precos = spider.get_prices()
        
        dispatcher.utter_message(text=f"{precos}")
        
        return []