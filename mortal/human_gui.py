#!/usr/bin/env python3
"""tkinter GUI for playing a mahjong hanchan as a human against the Mortal AI.

The game loop runs in a background thread (Rust arena via `human_gui_vs_py`);
`HumanGuiEngine.react_state` is called on each human turn, which pushes the
state to the GUI thread and blocks until the user picks an action.
"""
import os
import sys
import queue
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'log-viewer', 'files', 'images'
)
POSE_UPRIGHT = 1
POSE_LAID = 3
TILE_ZOOM = 2

HONOR_IMG = {
    'E': 'ji_e', 'S': 'ji_s', 'W': 'ji_w', 'N': 'ji_n',
    'P': 'no', 'F': 'ji_h', 'C': 'ji_c',
}
BAKAZE_STR = {'E': '东', 'S': '南', 'W': '西', 'N': '北'}
KIND_STR = {'E': '东', 'S': '南', 'W': '西', 'N': '北', 'P': '中', 'F': '发', 'C': '白'}

ACTIONS = [
    ('pon', '碰'), ('daiminkan', '大明杠'), ('kakan', '加杠'), ('ankan', '暗杠'),
    ('chi_l', '吃左'), ('chi_m', '吃中'), ('chi_h', '吃右'),
    ('riichi', '立直'), ('tsumo', '自摸'), ('ron', '荣和'),
    ('ryukyoku', '流局'), ('pass', '过'),
]
CAN_ACTION = {
    'pon': 'can_pon', 'daiminkan': 'can_daiminkan', 'kakan': 'can_kakan',
    'ankan': 'can_ankan', 'chi_l': 'can_chi_low', 'chi_m': 'can_chi_mid',
    'chi_h': 'can_chi_high', 'riichi': 'can_riichi', 'tsumo': 'can_tsumo_agari',
    'ron': 'can_ron_agari', 'ryukyoku': 'can_ryukyoku',
}


def tile_img_name(tile):
    if tile in HONOR_IMG:
        return HONOR_IMG[tile]
    return f"{tile[1]}s{tile[0]}"


class TileImages:
    def __init__(self, root):
        self.root = root
        self._cache = {}

    def get(self, tile, pose=POSE_UPRIGHT):
        key = (tile, pose)
        if key not in self._cache:
            name = tile_img_name(tile)
            path = os.path.join(IMAGE_DIR, f"p_{name}_{pose}.gif")
            if os.path.exists(path):
                img = tk.PhotoImage(file=path)
                if TILE_ZOOM > 1:
                    img = img.zoom(TILE_ZOOM, TILE_ZOOM)
                self._cache[key] = img
            else:
                self._cache[key] = None
        return self._cache[key]


class HumanGuiEngine:
    engine_type = 'human-gui'
    name = 'human'

    def __init__(self, states_queue, result_queue):
        self.states_queue = states_queue
        self.result_queue = result_queue

    def react_state(self, state):
        self.states_queue.put(state)
        return self.result_queue.get()

    def start_game(self, game_idx):
        pass

    def end_kyoku(self, game_idx):
        pass

    def end_game(self, game_idx, scores):
        pass


class HumanGuiApp:
    def __init__(self, root, states_queue, result_queue, done_event, scores_queue):
        self.root = root
        self.images = TileImages(root)
        self.states_queue = states_queue
        self.result_queue = result_queue
        self.done_event = done_event
        self.scores_queue = scores_queue
        self.state = None

        root.title('人类 vs Mortal AI')
        root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.top_info = tk.Label(root, text='', font=('TkDefaultFont', 12, 'bold'))
        self.top_info.pack()

        self.board = tk.Frame(root)
        self.board.pack(fill=tk.BOTH, expand=True)
        self.board.grid_rowconfigure(0, weight=1)
        self.board.grid_rowconfigure(1, weight=5)
        self.board.grid_rowconfigure(2, weight=1)
        self.board.grid_rowconfigure(3, weight=0)
        self.board.grid_columnconfigure(0, weight=1)
        self.board.grid_columnconfigure(2, weight=1)

        # top opponent (seat 2)
        self.top_frame = self._player_frame(self.board, 'top', 0)
        self.board.grid_columnconfigure(1, weight=3)
        # middle: left opponent (seat 3) | center info | right opponent (seat 1)
        self.left_frame = self._player_frame(self.board, 'left', 1)
        self.center = self._center_frame(self.board)
        self.right_frame = self._player_frame(self.board, 'right', 2)
        # bottom: local player's river (seat 0)
        self.bottom_frame = self._player_frame(self.board, 'bottom', 3)
        # bottom center: your hand + action buttons
        self.action = self._action_frame(self.board)

        # 4 corners hold each player's melds (副露), on their right-hand side
        self.corner_of_seat = {0: 'br', 1: 'tr', 2: 'tl', 3: 'bl'}
        self.corners = {}
        for key, (r, c) in {'tl': (0, 0), 'tr': (0, 2), 'bl': (2, 0), 'br': (2, 2)}.items():
            corner = self._corner_frame(self.board)
            corner['frame'].grid(row=r, column=c, sticky='nsew')
            self.corners[key] = corner

        self.top_frame['frame'].grid(row=0, column=1, sticky='nsew')
        self.left_frame['frame'].grid(row=1, column=0, sticky='nsew')
        self.center['frame'].grid(row=1, column=1, sticky='nsew')
        self.right_frame['frame'].grid(row=1, column=2, sticky='nsew')
        self.bottom_frame['frame'].grid(row=2, column=1, sticky='nsew')
        self.action['frame'].grid(row=3, column=0, columnspan=3, sticky='s')

        self.result_label = tk.Label(root, text='', font=('TkDefaultFont', 11, 'bold'))
        self.result_label.pack()

        self.waiting_action = False
        self.root.after(30, self.poll)
        self.root.after(200, self.check_done)

    # ---------- widget construction ----------

    def _player_frame(self, parent, position, _idx):
        f = tk.Frame(parent, bd=1, relief=tk.RIDGE, padx=4, pady=4)
        name = tk.Label(f, text='', font=('TkDefaultFont', 10, 'bold'))
        name.pack()
        score = tk.Label(f, text='')
        score.pack()
        river = tk.Frame(f)
        river.pack()
        return {'frame': f, 'name': name, 'score': score, 'river_frame': river}

    def _corner_frame(self, parent):
        f = tk.Frame(parent, bd=1, relief=tk.RIDGE, padx=4, pady=4)
        label = tk.Label(f, text='', font=('TkDefaultFont', 9))
        label.pack(anchor='w')
        melds = tk.Frame(f)
        melds.pack(anchor='w')
        return {'frame': f, 'label': label, 'melds': melds}

    def _center_frame(self, parent):
        f = tk.Frame(parent, padx=6, pady=6)
        status = tk.Label(f, text='', justify=tk.LEFT, font=('TkDefaultFont', 11))
        status.pack()
        ld = tk.Frame(f)
        ld.pack(pady=4)
        tk.Label(ld, text='最近打出:', font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        last_tile = tk.Label(ld)
        last_tile.pack(side=tk.LEFT)
        return {'frame': f, 'status': status, 'last_discard': last_tile}

    def _action_frame(self, parent):
        f = tk.Frame(parent, padx=6, pady=6)
        hand = tk.Frame(f, bd=2, relief=tk.SOLID, padx=6, pady=6)
        hand.pack(pady=4)
        btns = tk.Frame(f)
        btns.pack()
        buttons = {}
        for action, text in ACTIONS:
            b = tk.Button(btns, text=text, state=tk.DISABLED,
                          command=lambda a=action: self.on_button(a))
            b.grid(row=len(buttons) // 5, column=len(buttons) % 5, padx=2, pady=2)
            buttons[action] = b
        return {'frame': f, 'hand_frame': hand, 'hand': [], 'buttons': buttons}

    # ---------- event handling ----------

    def poll(self):
        try:
            while True:
                state = self.states_queue.get_nowait()
                self.state = state
                self.waiting_action = True
                print(f"[gui] render# tehai={state['tehai']} last_tsumo={state['last_self_tsumo']} "
                      f"cans_discard={state['cans']['can_discard']} cans_pon={state['cans']['can_pon']}",
                      file=sys.stderr)
                try:
                    self.render(state)
                except Exception as ex:
                    print(f'[gui] render 异常: {ex}', file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def check_done(self):
        if self.done_event.is_set():
            print('[gui] 游戏线程已结束', file=sys.stderr)
            scores = None
            try:
                scores = self.scores_queue.get_nowait()
            except queue.Empty:
                pass
            if isinstance(scores, Exception):
                print(f'[gui] 游戏线程异常: {scores}', file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.result_label.config(text=f'游戏出错: {scores}')
            elif scores is not None:
                print('[gui] 显示结果', file=sys.stderr)
                self.show_result(scores)
                self._disable_all()
        else:
            self.root.after(200, self.check_done)

    def _disable_all(self):
        for pos in ('top', 'left', 'right', 'bottom'):
            frame = getattr(self, pos + '_frame')
            for lab in frame['river_frame'].winfo_children():
                pass
        for lab in self.action['hand_frame'].winfo_children():
            lab.unbind('<Double-Button-1>')
        for b in self.action['buttons'].values():
            b.config(state=tk.DISABLED)

    def on_button(self, action):
        if self.waiting_action:
            self.waiting_action = False
            try:
                self.result_queue.put_nowait(action)
                print(f'[gui] 已执行: {action}', file=sys.stderr)
            except queue.Full:
                self.waiting_action = True
                print(f'[gui] 点击 {action} 但队列满(忽略)', file=sys.stderr)
        else:
            print(f'[gui] 点击 {action} 但非等待动作(忽略)', file=sys.stderr)

    def on_tile_discard(self, tile):
        can_discard = self.state is not None and self.state['cans'].get('can_discard', False)
        in_hand = self.state is not None and tile in self.state['tehai']
        if self.waiting_action and can_discard and in_hand:
            self.waiting_action = False
            try:
                self.result_queue.put_nowait(tile)
                print(f'[gui] 已执行: 弃 {tile}', file=sys.stderr)
            except queue.Full:
                self.waiting_action = True
                print(f'[gui] 弃 {tile} 但队列满(忽略)', file=sys.stderr)
        else:
            print(f'[gui] 双击 {tile} 被忽略:can_discard={can_discard} in_hand={in_hand}', file=sys.stderr)

    def on_close(self):
        # unblock a pending react_state if the game is still running
        try:
            self.result_queue.put_nowait('__QUIT__')
        except queue.Full:
            pass
        self.root.destroy()

    # ---------- rendering ----------

    def render(self, state):
        self.state = state
        player_id = state['player_id']
        seats = {
            'bottom': player_id,
            'right': (player_id + 1) % 4,
            'top': (player_id + 2) % 4,
            'left': (player_id + 3) % 4,
        }
        pose_map = {'bottom': POSE_UPRIGHT, 'top': POSE_UPRIGHT,
                    'right': POSE_UPRIGHT, 'left': POSE_UPRIGHT}

        names = ['你', 'Mortal', 'Mortal', 'Mortal']
        for pos, seat in seats.items():
            frame = getattr(self, pos + '_frame')
            frame['name'].config(text=f"{names[seat]} (座{seat})")
            frame['score'].config(text=f"{state['scores'][seat]}")
            kawa = state['kawa'][seat]
            pose = pose_map[pos]
            river_f = frame['river_frame']
            for child in river_f.winfo_children():
                child.destroy()
            n_tiles = len(kawa)
            for idx in range(n_tiles):
                tile = kawa[idx]
                img = self.images.get(tile, pose)
                lab = tk.Label(river_f)
                lab.config(image=img)
                lab.image = img
                # 四家牌河统一:从左到右,每行 7 张,新行排下面
                lab.grid(row=idx // 7, column=idx % 7)

        # 副露:每家显示在其右手边的角落
        for seat in range(4):
            corner = self.corners[self.corner_of_seat[seat]]
            corner['label'].config(text=f'{names[seat]} (座{seat}) 副露')
            melds_f = corner['melds']
            for child in melds_f.winfo_children():
                child.destroy()
            for mi, mel in enumerate(state['fuuro'][seat]):
                for t in mel:
                    img = self.images.get(t)
                    lab = tk.Label(melds_f)
                    lab.config(image=img)
                    lab.image = img
                    lab.pack(side=tk.LEFT)
                if mi < len(state['fuuro'][seat]) - 1:
                    tk.Label(melds_f, text=' ').pack(side=tk.LEFT)

        center = self.center
        riichi = ''.join('立' if state['riichi_declared'][s] else '-' for s in range(4))
        cans = state['cans']
        if not cans.get('can_discard', False):
            hint = '轮到决策:'
            for action, key in CAN_ACTION.items():
                if cans.get(key):
                    hint += f' {dict(ACTIONS)[action]}'
            hint += ' 过' if (cans.get('can_chi_low') or cans.get('can_chi_mid')
                              or cans.get('can_chi_high') or cans.get('can_pon')
                              or cans.get('can_daiminkan') or cans.get('can_ron_agari')) else ''
            hint += '(不能打牌,请点按钮)'
        else:
            hint = '双击手牌弃牌'
        status = (f"{BAKAZE_STR[state['bakaze']]} {state['kyoku']}局 "
                  f"{state['honba']}本场 供托{state['kyotaku']} 余牌{state['tiles_left']} "
                  f"向听{state['shanten']} 自摸:{state['last_self_tsumo'] or '-'} "
                  f"立直:{riichi}\n{hint}")
        center['status'].config(text=status)

        lt = state.get('last_kawa_tile')
        if lt:
            img = self.images.get(lt)
            center['last_discard'].config(image=img)
            center['last_discard'].image = img
        else:
            center['last_discard'].config(image=None)

        action = self.action
        tehai = state['tehai']
        drawn = state['last_self_tsumo']
        discardable = cans.get('can_discard', False)
        hand_frame = action['hand_frame']
        for child in hand_frame.winfo_children():
            child.destroy()
        action['hand'] = []
        for i, tile in enumerate(tehai):
            lab = tk.Label(hand_frame, cursor='hand2' if discardable else 'arrow')
            img = self.images.get(tile)
            lab.config(image=img)
            lab.image = img
            if tile == drawn:
                lab.config(highlightbackground='#e11', highlightcolor='#e11', highlightthickness=3)
            lab.bind('<Double-Button-1>', lambda e, t=tile: self.on_tile_discard(t))
            lab.grid(row=0, column=i)
            action['hand'].append(lab)

        for action_key, key in CAN_ACTION.items():
            state_ = tk.NORMAL if cans.get(key) else tk.DISABLED
            action['buttons'][action_key].config(state=state_)
        can_pass = (cans.get('can_chi_low') or cans.get('can_chi_mid')
                    or cans.get('can_chi_high') or cans.get('can_pon')
                    or cans.get('can_daiminkan') or cans.get('can_ron_agari'))
        action['buttons']['pass'].config(state=tk.NORMAL if can_pass else tk.DISABLED)

    def show_result(self, scores):
        self.root.title('游戏结束')
        lines = ['===== 游戏结束 · 结果(净分差 = 终局分 - 25000) =====']
        total = 0
        for i, s in enumerate(scores, 1):
            total += s[0]
            lines.append(f"半庄{i}: 你(东家) {s[0]:+d} | Mortal {s[1]:+d} / {s[2]:+d} / {s[3]:+d}")
        lines.append(f"你的总净分: {total:+d}")
        s = scores[0]
        winner = max(range(4), key=lambda k: s[k])
        if s[winner] > 0:
            lines.append(('你 和牌结束!' if winner == 0 else f'Mortal(座{winner}) 和牌结束'))
        else:
            lines.append('流局')
        lines.append('(游戏已结束,可关闭窗口)')
        self.result_label.config(text='\n'.join(lines),
                                 font=('TkDefaultFont', 13, 'bold'), fg='#c00')


def run_gui(engine, log_dir, seed_start, state_file):
    """Run the game with a tkinter GUI. Returns the per-game net score deltas
    (list of [human, mortal, mortal, mortal]) after the window is closed."""
    from libriichi.arena import OneVsThree

    states_queue = queue.Queue()
    result_queue = queue.Queue(maxsize=1)
    done_event = threading.Event()
    scores_queue = queue.Queue()
    human = HumanGuiEngine(states_queue, result_queue)

    root = tk.Tk()
    app = HumanGuiApp(root, states_queue, result_queue, done_event, scores_queue)
    root.geometry('1600x1400')

    def game_thread():
        try:
            env = OneVsThree(disable_progress_bar=True, log_dir=log_dir)
            scores = env.human_gui_vs_py(
                human=human, engine=engine,
                seed_start=seed_start, seed_count=1,
            )
            scores_queue.put(scores)
            print(f'[gui] 游戏完成, scores={scores}', file=sys.stderr)
        except Exception as ex:
            print(f'[gui] 游戏线程异常: {ex}', file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            scores_queue.put(ex)
        finally:
            done_event.set()

    threading.Thread(target=game_thread, daemon=True).start()
    root.mainloop()
    if scores_queue.empty():
        return None
    scores = scores_queue.get()
    if isinstance(scores, Exception):
        raise scores
    return scores
