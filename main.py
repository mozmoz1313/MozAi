import threading
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.progressbar import ProgressBar
from kivy.animation import Animation
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from brain import train_bot, predict, backtest

Window.clearcolor = (0.05, 0.05, 0.1, 1)

# ============================================================
#  SPLASH SCREEN
# ============================================================
class SplashScreen(Screen):
    def on_enter(self):
        layout = BoxLayout(orientation='vertical')
        self.logo = Image(source='MozStudio.png', opacity=0,
                          size_hint=(1, 0.7))
        tagline = Label(
            text='[b]MozAi — Scalper Élite[/b]',
            markup=True,
            font_size='18sp',
            color=(0.4, 0.8, 1, 0),
            size_hint=(1, 0.3)
        )
        layout.add_widget(self.logo)
        layout.add_widget(tagline)
        self.add_widget(layout)

        anim_logo = (Animation(opacity=1, duration=1.5) +
                     Animation(opacity=1, duration=1.0) +
                     Animation(opacity=0, duration=1.0))
        anim_text = Animation(color=(0.4, 0.8, 1, 1), duration=1.5)
        anim_logo.bind(on_complete=lambda *a:
                       setattr(self.manager, 'current', 'main'))
        anim_logo.start(self.logo)
        anim_text.start(tagline)


# ============================================================
#  DASHBOARD PRINCIPAL
# ============================================================
class MainConsole(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model      = None
        self.is_running = False
        self._build_ui()

    def _build_ui(self):
        root = BoxLayout(orientation='vertical', padding=12, spacing=8)

        # --- En-tête ---
        header = Label(
            text='[b]🤖 MozAi — XAUUSD Scalper[/b]',
            markup=True,
            font_size='20sp',
            color=(0.4, 0.9, 1, 1),
            size_hint=(1, 0.08)
        )

        # --- Zone de log scrollable ---
        scroll = ScrollView(size_hint=(1, 0.5))
        self.log_label = Label(
            text='Bienvenue dans MozAi.\nCharge un modèle ou lance l\'Éducation.',
            markup=True,
            font_size='13sp',
            color=(0.85, 0.95, 1, 1),
            size_hint_y=None,
            text_size=(Window.width - 30, None),
            halign='left',
            valign='top'
        )
        self.log_label.bind(texture_size=self.log_label.setter('size'))
        scroll.add_widget(self.log_label)

        # --- Barre de progression ---
        self.progress = ProgressBar(max=100, value=0, size_hint=(1, 0.04))

        # --- Signal actuel ---
        self.signal_label = Label(
            text='⏳ En attente d\'analyse...',
            font_size='16sp',
            bold=True,
            color=(1, 1, 0.4, 1),
            size_hint=(1, 0.08)
        )

        # --- Boutons ---
        btn_layout = BoxLayout(
            orientation='horizontal',
            spacing=8,
            size_hint=(1, 0.12)
        )

        self.btn_train   = self._btn('🎓 Éducation',  (0.2, 0.6, 0.9, 1), self.start_training)
        self.btn_predict = self._btn('🔍 Analyser',   (0.2, 0.8, 0.4, 1), self.start_predict)
        self.btn_back    = self._btn('🔄 Backtest',   (0.7, 0.5, 0.9, 1), self.start_backtest)
        self.btn_auto    = self._btn('🚀 Auto ON/OFF',(0.9, 0.6, 0.2, 1), self.toggle_auto)

        for b in [self.btn_train, self.btn_predict,
                  self.btn_back, self.btn_auto]:
            btn_layout.add_widget(b)

        # --- Mémoire ---
        self.mem_label = Label(
            text='🧠 Mémoire : chargement...',
            font_size='11sp',
            color=(0.5, 0.8, 0.5, 1),
            size_hint=(1, 0.06)
        )

        root.add_widget(header)
        root.add_widget(scroll)
        root.add_widget(self.progress)
        root.add_widget(self.signal_label)
        root.add_widget(btn_layout)
        root.add_widget(self.mem_label)
        self.add_widget(root)

        Clock.schedule_once(self._load_memory_display, 1)

    def _btn(self, text, color, callback):
        b = Button(
            text=text,
            font_size='12sp',
            background_color=color,
            background_normal=''
        )
        b.bind(on_press=callback)
        return b

    # ---- Affichage mémoire ----
    def _load_memory_display(self, dt):
        import json, os
        path = 'mozai_memory.json'
        if os.path.exists(path):
            with open(path) as f:
                mem = json.load(f)
            self.mem_label.text = (
                f"🧠 Sessions: {mem.get('sessions',0)} | "
                f"Epochs: {mem.get('total_epochs',0)} | "
                f"Best loss: {mem.get('best_loss',9999):.4f}"
            )

    # ---- Log thread-safe ----
    def _log(self, msg):
        def _update(dt):
            self.log_label.text += f'\n{msg}'
        Clock.schedule_once(_update, 0)

    def _set_progress(self, val):
        def _update(dt):
            self.progress.value = val
        Clock.schedule_once(_update, 0)

    # ---- Éducation ----
    def start_training(self, *a):
        if self.is_running:
            return
        self.is_running = True
        self.btn_train.disabled = True
        self._log('━━━━━━━━━━━━━━━━━━━━')
        self._log('🎓 Éducation démarrée...')
        self._set_progress(5)

        def _run():
            epoch_tracker = [0]
            def cb(msg):
                self._log(msg)
                epoch_tracker[0] += 1
                self._set_progress(min(95, epoch_tracker[0] * 2))
            self.model = train_bot(epochs=300, update_callback=cb)
            self._set_progress(100)
            Clock.schedule_once(lambda dt: self._load_memory_display(dt), 0.5)
            self.is_running = False
            Clock.schedule_once(lambda dt: setattr(self.btn_train, 'disabled', False), 0)

        threading.Thread(target=_run, daemon=True).start()

    # ---- Analyse ----
    def start_predict(self, *a):
        if self.is_running:
            return
        self.is_running = True
        self._log('━━━━━━━━━━━━━━━━━━━━')
        self._log('🔍 Analyse en cours...')

        def _run():
            result = predict(self.model)
            self._log(result)
            Clock.schedule_once(
                lambda dt: setattr(self.signal_label, 'text', result.split('\n')[4]), 0)
            self.is_running = False

        threading.Thread(target=_run, daemon=True).start()

    # ---- Backtest ----
    def start_backtest(self, *a):
        if self.is_running:
            return
        self.is_running = True
        self._log('━━━━━━━━━━━━━━━━━━━━')

        def _run():
            result = backtest(self._log)
            self._log(result)
            self.is_running = False

        threading.Thread(target=_run, daemon=True).start()

    # ---- Mode automatique ----
    def toggle_auto(self, *a):
        if not hasattr(self, '_auto_event') or self._auto_event is None:
            self._auto_event = Clock.schedule_interval(
                lambda dt: self.start_predict(), 60)
            self.btn_auto.text = '🛑 Auto OFF'
            self._log('🚀 Mode automatique activé (60s)')
        else:
            self._auto_event.cancel()
            self._auto_event = None
            self.btn_auto.text = '🚀 Auto ON/OFF'
            self._log('⏹️ Mode automatique désactivé')


# ============================================================
#  APP PRINCIPALE
# ============================================================
class MozAiApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(SplashScreen(name='splash'))
        sm.add_widget(MainConsole(name='main'))
        sm.current = 'splash'
        return sm


if __name__ == '__main__':
    MozAiApp().run()
