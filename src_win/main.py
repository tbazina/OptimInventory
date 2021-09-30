import os
import sys
import gc
import numpy as np
from expsimnb import findOptExp
import time
import multiprocessing as mp
import smtplib
from email.message import EmailMessage
import datetime
import ntplib
import ipgetter

os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'multisamples', '0')
from kivy.app import App
from kivy.lang.builder import Builder
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.garden.filebrowser import FileBrowser

"""
------------------------- Ulazni Podaci -------------------------
"""

# Ulazni folder
# ulFol = 'input_folder'

# Izlazni folder
# izlFol = 'output_folder'

# Fileovi za analiziranje
# files = []
# with os.scandir(path=ulFol) as it:
#     for entry in it:
#         if entry.name.endswith('.csv') and entry.is_file():
#             files.append((
#                 entry.name[:-4],
#                 np.genfromtxt(
#                     entry.path, delimiter=';', dtype=np.str_
#                 )
#             ))


# Pocetna razina zaliha
# pocZalExp = 0  # postaviti na nulu ako je jednaka max razini zaliha

# Maksimalna razina zaliha (gornja i donja)
# - dvije opcije: gornja i donja granica, konstantna
# maxZalD = 20  # Postaviti jednaku vrijednost kao maxZalG ako je konstantna
# maxZalG = 10000  # Postaviti jednaku vrijednost kao maxZalD ako je konstantna

# Minimalna razina zaliha (gornja i donja)
# - tri opcije: donja i gornja granica, donja i max razina zaliha, konstantna
# Postaviti minZalD = minZalG ako je konstantna
# Postaviti minZalG >= maxZalG ako je donja i max razina zaliha
# minZalD = 5
# minZalG = 10000

# Minimalna veličina narudžbe
# minNar = 7

# Dani (takt) dobave
# daniDob = 9

# Radno vrijeme dobavljaca i subjekta
# radVrDob = 5
# radVrSub = 7

# Način provjere Backlogginga
# - za provjeru BL-a postaviti blChk = 0
# - ako nije potrebno provjeriti BL postaviti blChk = 1
# blChk = 1

# Postotak Backlogging-a (donja i gornja granica)
# blDPerc = 0.
# blGPerc = 30.

# Postotak zadovoljavanja potraznje trzista po proizvodima
# (gornja i donja vrijednost)
# pzptProDPerc = 90.
# pzptProGPerc = 95.

# Postotak zadovoljavanja potraznje trzista po proizvodima
# (gornja i donja vrijednost)
# pzptDanDPerc = 89.
# pzptDanGPerc = 90.

# Način provjere PZPT
# - za provjeru PZPT-a postaviti pzptXXXChk = 0
# - ako nije potrebno provjeriti odabrani PZPT postaviti pzptXXXChk = 1
# pzptProChk = 0
# pzptDanChk = 0

# Provjera velicine dobave
# Ako se provjerava postaviti velDobChk = 0
# velDobChk = 0
# Gornja i donja granica velicine dobave
# velDobD = 500
# velDobG = 2000

# Spremiti koliko najboljih eksperimenata
# sprNaj = 1

# Način pretraživanja područja
# - za Brute-Force pretraživanje postaviti algType = 0
# - za Dijagonalno + Brute-Force pretraživanje postaviti algType = 1
# algType = 1

# Kriterij zaustavljanja dijagonalnog i početak Brute-Force pretraživanja
# - postotak od traženog PZPT
# diagStopPerc = 0.001

# Broj CPU
# max_proc = mp.cpu_count()
# proc_num = 4

# Email report
# email_chk = False
# recipient = 'example@gmail.com'
# sender = 'ttest1991.1234@gmail.com'
# smtp_server = 'smtp.gmail.com'
# password = 'Test123test'

"""
------------------------------------------------------------------------------
"""

"""
------------------------- Objekti optimizacije -------------------------
"""

class OIClass(object):
    def __init__(self):
        self.error_msgs = []
        return

    def get_error_msgs(self):
        return self.error_msgs

    def clean_error_msgs(self):
        self.error_msgs = []
        return

    def setParameters(
            self, ulFol, izlFol, pocZalExp, maxZalD, maxZalG, minZalD, minZalG,
            minNar, daniDob, radVrDob, radVrSub, blChk, blDPerc, blGPerc,
            pzptProDPerc, pzptProGPerc, pzptDanDPerc, pzptDanGPerc,
            pzptProChk, pzptDanChk, velDobChk, velDobD, velDobG, sprNaj,
            diagStopPerc, max_proc, proc_num, email_chk, recipient, sender,
            smtp_server, password
    ):

        # Provjera ulazne vrijednosti ne-negativne
        for i in (pocZalExp, maxZalD, maxZalG, minZalD, minZalG, minNar,
                  daniDob, radVrDob, radVrSub, blDPerc, blGPerc, pzptProDPerc,
                  pzptProGPerc, pzptDanDPerc, pzptDanGPerc, velDobD, velDobG,
                  sprNaj, diagStopPerc, proc_num):
            if i < 0:
                self.error_msgs.append(
                    'Use only non-negative values! {}'.format(i)
                )
                return

        self.ulFol = ulFol
        # Error ulaz folder ili file
        if (not os.path.isdir(ulFol)) and (not os.path.isfile(ulFol)):
            self.error_msgs.append(
                'Load Files Error: Not a folder / file!'
            )
            return
        self.izlFol = izlFol
        # Error izlaz folder
        if not os.path.isdir(izlFol):
            self.error_msgs.append(
                'Output Directory Error: Not a folder!'
            )
            return
        self.pocZalExp = pocZalExp
        self.maxZalD = maxZalD
        self.maxZalG = maxZalG
        # Error maxZalD > maxZalG
        if maxZalD > maxZalG:
            self.error_msgs.append(
                'Order-up-to level Error: {} > {}'.format(maxZalD, maxZalG)
            )
            return
        self.minZalD = minZalD
        self.minZalG = minZalG
        # Error minZalD > minZalG
        if minZalD > minZalG:
            self.error_msgs.append(
                'Reorder-point level Error: {} > {}'.format(minZalD, minZalG)
            )
            return
        self.minNar = minNar
        self.daniDob = daniDob
        self.radVrDob = radVrDob
        self.radVrSub = radVrSub
        # Error radVrDob > radVrSub
        if radVrDob > radVrSub:
            self.error_msgs.append(
                'Supplier\'s schedule Error: {} > {}'.format(radVrDob, radVrSub)
            )
            return
        self.blChk = blChk
        self.blDPerc = blDPerc
        self.blGPerc = blGPerc
        if not blChk:
            # Error blDPerc > blGPerc
            if blDPerc > blGPerc:
                self.error_msgs.append(
                    'Backlog Error: {} > {}'.format(blDPerc, blGPerc)
                )
                return
            # Error blDPerc > 100
            if blDPerc > 100:
                self.error_msgs.append(
                    'Backlog Error: {} > {}'.format(blDPerc, 100)
                )
                return
            # Error blGPerc > 100
            if blGPerc > 100:
                self.error_msgs.append(
                    'Backlog Error: {} > {}'.format(blGPerc, 100)
                )
                return
        self.pzptProDPerc = pzptProDPerc
        self.pzptProGPerc = pzptProGPerc
        if not pzptProChk:
            # Error pzptProDPerc  > pzptProGPerc
            if pzptProDPerc > pzptProGPerc:
                self.error_msgs.append(
                    'Fill rate Error: {} > {}'.format(pzptProDPerc, pzptProGPerc)
                )
                return
            # Error pzptProDPerc > 200
            if pzptProDPerc > 200:
                self.error_msgs.append(
                    'Fill rate Error: {} > {}'.format(pzptProDPerc, 200)
                )
                return
            # Error pzptProGPerc > 200
            if pzptProGPerc > 200:
                self.error_msgs.append(
                    'Fill rate Error: {} > {}'.format(pzptProGPerc, 200)
                )
                return
        self.pzptDanDPerc = pzptDanDPerc
        self.pzptDanGPerc = pzptDanGPerc
        if not pzptDanChk:
            # Error pzptDanDPerc   > pzptDanGPerc
            if pzptDanDPerc > pzptDanGPerc:
                self.error_msgs.append(
                    'Fill rate Error: {} > {} '.format(pzptDanDPerc, pzptDanGPerc)
                )
                return
            # Error pzptDanDPerc > 100
            if pzptDanDPerc > 100:
                self.error_msgs.append(
                    'Fill rate Error: {} > {}'.format(pzptDanDPerc, 100)
                )
                return
            # Error pzptDanGPerc > 100
            if pzptDanGPerc > 100:
                self.error_msgs.append(
                    'Fill rate Error: {} > {}'.format(pzptDanGPerc, 100)
                )
                return
        self.pzptProChk = pzptProChk
        self.pzptDanChk = pzptDanChk
        self.velDobChk = velDobChk
        self.velDobD = velDobD
        self.velDobG = velDobG
        if not velDobChk:
            # Error velDobD > velDobG
            if velDobD > velDobG:
                self.error_msgs.append(
                    'Shipment size Error: {} > {}'.format(velDobD, velDobG)
                )
                return
        self.sprNaj = sprNaj
        self.diagStopPerc = diagStopPerc
        # Error diagStopPerc too high
        if not pzptProChk:
            if diagStopPerc > pzptProDPerc:
                self.error_msgs.append(
                    'Fine search tolerance Error: {} > {}'.format(
                        diagStopPerc, pzptProDPerc
                    )
                )
                return
        if not pzptDanChk:
            if diagStopPerc > pzptDanDPerc:
                self.error_msgs.append(
                    'Fine search tolerance Error: {} > {}'.format(
                        diagStopPerc, pzptDanDPerc
                    )
                )
                return
        self.max_proc = max_proc
        self.proc_num = proc_num
        self.email_chk = email_chk
        self.recipient = recipient
        self.sender = sender
        self.smtp_server = smtp_server
        self.password = password
        # Error email text empty
        if email_chk:
            if not recipient:
                self.error_msgs.append('Send email Error: Email address empty')
                return
            if not sender:
                self.error_msgs.append('Send email Error: Email from empty')
                return
            if not smtp_server:
                self.error_msgs.append('Send email Error: SMTP server empty')
                return
            if not password:
                self.error_msgs.append('Send email Error: Password empty')
                return

        # Lista (ime, np.array) ulaznih datoteka
        self.files = []
        if os.path.isdir(self.ulFol):
            with os.scandir(path=self.ulFol) as it:
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        self.files.append((
                            entry.name[:-4],
                            np.genfromtxt(
                                entry.path, delimiter=';', dtype=np.str_
                            )
                        ))
        elif os.path.isfile(self.ulFol):
            with open(self.ulFol, mode='r') as entry:
                if entry.name.endswith('.csv'):
                    self.files.append((
                        os.path.basename(entry.name)[:-4],
                        np.genfromtxt(
                            entry.name, delimiter=';', dtype=np.str_
                        )
                    ))

        # Provjera ima li ulaznih datoteka
        if not self.files:
            self.error_msgs.append(
                'Input file/folder Error: There is no input files!')
            return

        # Prosjecna velicina ulaznih datoteka
        self.avg_size = int(
            np.array([f[1].shape[0] for f in self.files]).mean()
        )

        # Argumenti funkcije potrebni za provodenje eksperimenta
        self.exp_args = [
            pocZalExp, maxZalD, maxZalG, minZalD, minZalG, minNar, daniDob,
            radVrDob, radVrSub, blChk, blDPerc, blGPerc, pzptProDPerc,
            pzptProGPerc, pzptDanDPerc, pzptDanGPerc, pzptProChk, pzptDanChk,
            izlFol, sprNaj, diagStopPerc, velDobChk, velDobD, velDobG
        ]

    def startTimeCount(self):
        """
        Pocetni datum i vrijeme pohranjeni u atribut

        :return: None
        """
        self.start_time = datetime.datetime.now()

    def evaluateParallel(self, findOptExp):
        """
        Paralelno pokretanje SE

        :param findOptExp:
        Funkcija za samostalno trazenje i spremanje optimalnih SE

        :return:
        None
        """
        # Multiprocessing Pool
        self.brojaciZadExp = []
        self.pool = mp.Pool(processes=self.proc_num)
        for f in self.files:
            old_name = f[0] + '_'
            if f[1].ndim == 1:
                potr = f[1].astype(np.int64)
                prepend = np.array([], dtype=np.str_)
            else:
                potr = f[1][:, -1].astype(np.int64)
                # Empty column before new experiment
                empty_col = np.array(
                    ['' for _ in range(f[1].shape[0])], dtype=np.str_
                )
                # Old experiments at beginning of the file
                prepend = np.concatenate(
                    (f[1], empty_col[:, np.newaxis]), axis=-1
                )
            # print(f[0])
            self.brojaciZadExp.append(self.pool.apply_async(
                func=findOptExp,
                args=[potr, old_name, prepend] + self.exp_args
            ))
        self.pool.close()

    def stopParallelProcesses(self):
        """
        Stop multiprocessing.Pool with terminate

        :return:
        None
        """
        self.pool.terminate()
        self.pool.join()

    def endTimeCount(self):
        """
        Vrijeme potrebno za zavrsetak evih simulacijskih eksperimenata.
        Brojac izlaznih datoteka
        Cleanup multiprocessing.Pool

        :return:
        None
        """
        self.end_time = datetime.datetime.now()
        self.exp_time = self.end_time - self.start_time
        # Brojac izlaznih datoteka
        self.output_no = sum([br.get() for br in self.brojaciZadExp])
        self.pool.join()
        return

    def checkIfReady(self):
        """
        Check if Pool processes have finished and result is accessible

        :return:
        None
        """
        return all([br.ready() for br in self.brojaciZadExp])

    def sendEmail(self):
        """
        Slanje mail obavijesti o zavrsetku eksperimenta

        :return:
        None
        """
        if self.email_chk:
            msg = EmailMessage()
            subject = 'OptimInventory v3.0 report'
            body = [
                "Project started on {:02d}.{:02d}.{}., {}".format(
                    self.start_time.day, self.start_time.month,
                    self.start_time.year, self.start_time.time()
                ),
                "IP address {}".format(ipgetter.myip()),
                "No. of input files {}".format(len(self.files)),
                "Average size of input file {} days".format(self.avg_size),
                "No. of output files {}".format(self.output_no),
                "No. of threads {}".format(self.proc_num),
                "Fine search tolerance {}%".format(self.diagStopPerc),
                "Project completed on {:02d}.{:02d}.{}., {}".format(
                    self.end_time.day, self.end_time.month, self.end_time.year,
                    self.end_time.time()
                ),
                "Total analysis time {} days {} h {} m {} s".format(
                    self.exp_time.days, self.exp_time.seconds // 3600,
                    self.exp_time.seconds // 60 % 60,
                    self.exp_time.total_seconds() % 60
                )
            ]
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = self.recipient
            msg.set_content("\n".join(body))

            with smtplib.SMTP_SSL(self.smtp_server, 465) as server:
                server.ehlo()
                server.login(user=self.sender, password=self.password)
                server.send_message(
                    msg=msg, from_addr=self.sender, to_addrs=self.recipient,
                )


"""
------------------------- Objekti GUI -------------------------
"""
if getattr(sys, 'frozen', False ) :
    bundle_dir = sys._MEIPASS
    Builder.load_file(''.join([bundle_dir, '\\optiminventory.kv']))


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    filter_function = ObjectProperty(None)


class ErrorContent(FloatLayout):
    close = ObjectProperty(None)
    err_msg = StringProperty('')


class OptimInventory(BoxLayout):
    text_id = StringProperty('')
    # START STOP button
    main_button = ObjectProperty()
    # Ulazne varijable
    ulFol = ObjectProperty()
    izlFol = ObjectProperty()
    pocZalExp = ObjectProperty()
    maxZalD = ObjectProperty()
    maxZalG = ObjectProperty()
    minZalD = ObjectProperty()
    minZalG = ObjectProperty()
    radVrDob = ObjectProperty()
    radVrSub = ObjectProperty()
    daniDob = ObjectProperty()
    minNar = ObjectProperty()
    pzptProChk = ObjectProperty()
    pzptProDPerc = ObjectProperty()
    pzptProGPerc = ObjectProperty()
    pzptDanChk = ObjectProperty()
    pzptDanDPerc = ObjectProperty()
    pzptDanGPerc = ObjectProperty()
    blChk = ObjectProperty()
    blDPerc = ObjectProperty()
    blGPerc = ObjectProperty()
    velDobChk = ObjectProperty()
    velDobD = ObjectProperty()
    velDobG = ObjectProperty()
    proc_num = ObjectProperty()
    diagStopPerc = ObjectProperty()
    email_chk = ObjectProperty()
    recipient = ObjectProperty()
    smtp_server = ObjectProperty()
    sender = ObjectProperty()
    password_snd = ObjectProperty()
    sprNaj = ObjectProperty()
    max_proc = NumericProperty(mp.cpu_count())

    def dismiss_popup(self):
        self._popup.clear_widgets()
        self._popup.dismiss()

    def show_load(self, text_id):
        content = LoadDialog(
            load=self.load, cancel=self.dismiss_popup)
        self.text_id = text_id
        if self.text_id == 'ulFol':
            title = 'Directory with input files / Single input file'
            content.filter_function = '*.csv'
        elif self.text_id == 'izlFol':
            title = 'Directory for output files'
            content.filter_function = self.is_dir
        self._popup = Popup(
            title=title, content=content, size_hint=(0.95, 0.95)
        )
        self._popup.open()

    def load(self, filename):
        self.ids[self.text_id].text = filename
        self._popup.clear_widgets()
        self.dismiss_popup()

    def is_dir(self, directory, filename):
        return os.path.isdir(os.path.join(directory, filename))

    def checkEmptyInt(self, text):
        if text == '':
            return 0
        else:
            return int(text)

    def checkEmptyFloat(self, text):
        if text == '':
            return 0.
        else:
            return float(text)

    def checkResult(self, dt):
        all_ready = OptimInv.checkIfReady()
        if all_ready:
            OptimInv.endTimeCount()
            OptimInv.sendEmail()
            self.main_button.text = 'START'
            gc.collect()
        return not all_ready

    def main_button_bind(self):
        if self.main_button.text == 'START':
            self.set_parameters()
            # Run simulation if there is no error messages
            if not OptimInv.get_error_msgs():
                self.main_button.text = 'STOP'
                OptimInv.startTimeCount()
                OptimInv.evaluateParallel(findOptExp)
                Clock.schedule_interval(self.checkResult, 0.5)
            else:
                # Popup containing error messages
                msg = ' '.join(err for err in OptimInv.get_error_msgs())
                content = ErrorContent(
                    close=self.dismiss_popup, err_msg=msg
                )
                title = 'Input Error'
                self._popup = Popup(
                    title=title, content=content, size_hint=(0.40, 0.40)
                )
                self._popup.open()
                # Clean error messages
                OptimInv.clean_error_msgs()

        elif self.main_button.text == 'STOP':
            self.main_button.text = 'START'
            OptimInv.stopParallelProcesses()
            gc.collect()

    def set_parameters(self):
        OptimInv.setParameters(
            self.ulFol.text,
            self.izlFol.text,
            int(self.pocZalExp.text),
            int(self.maxZalD.text),
            int(self.maxZalG.text),
            int(self.minZalD.text),
            int(self.minZalG.text),
            int(self.minNar.text),
            int(self.daniDob.text),
            int(self.radVrDob.text),
            int(self.radVrSub.text),
            1 - int(self.blChk.active),
            self.checkEmptyFloat(self.blDPerc.text),
            self.checkEmptyFloat(self.blGPerc.text),
            self.checkEmptyFloat(self.pzptProDPerc.text),
            self.checkEmptyFloat(self.pzptProGPerc.text),
            self.checkEmptyFloat(self.pzptDanDPerc.text),
            self.checkEmptyFloat(self.pzptDanGPerc.text),
            1 - int(self.pzptProChk.active),
            1 - int(self.pzptDanChk.active),
            1 - int(self.velDobChk.active),
            self.checkEmptyInt(self.velDobD.text),
            self.checkEmptyInt(self.velDobG.text),
            int(self.sprNaj.text),
            float(self.diagStopPerc.text),
            int(self.max_proc),
            int(self.proc_num.text),
            self.email_chk.active,
            self.recipient.text,
            self.sender.text,
            self.smtp_server.text,
            self.password_snd.text
        )


class OptimInventoryEnd(Label):
    pass


class OptimInventoryApp(App):
    def build(self):
        # strt = datetime.datetime(2018, 9, 1)
        # c = ntplib.NTPClient()
        # try:
        #     resp = c.request('europe.pool.ntp.org', version=3)
        #     now = datetime.datetime.fromtimestamp(resp.tx_time)
        # except:
        #     now = datetime.datetime.now()
        # if abs((now - strt).days) > 30:
        #     return OptimInventoryEnd()
        # else:
        return OptimInventory()


Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('ErrorContent', cls=ErrorContent)
"""
------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')
    OptimInv = OIClass()
    OptimInventoryApp().run()

