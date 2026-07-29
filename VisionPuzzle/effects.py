"""Lightweight visual effects — snap bursts, confetti, mode fades."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from VisionPuzzle import ui


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]
    radius: float = 3.0


@dataclass
class Effects:
    particles: list[Particle] = field(default_factory=list)
    rings: list[tuple[float, float, float, float, tuple[int, int, int]]] = field(default_factory=list)
    # x, y, radius, life, color
    fade: float = 0.0  # 0..1 white/dark overlay for transitions

    def burst(self, x: float, y: float, *, color=ui.SUCCESS, n: int = 18) -> None:
        for _ in range(n):
            ang = random.random() * math.tau
            spd = 2.5 + random.random() * 5.5
            self.particles.append(
                Particle(
                    x=x, y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd - 1.5,
                    life=1.0,
                    color=color,
                    radius=2.0 + random.random() * 2.5,
                )
            )
        self.rings.append((x, y, 8.0, 1.0, color))

    def confetti(self, w: int, h: int, n: int = 40) -> None:
        palette = [ui.ACCENT, ui.ACCENT_HOT, ui.SUCCESS, (220, 180, 140), (180, 160, 255)]
        for _ in range(n):
            self.particles.append(
                Particle(
                    x=random.uniform(0, w),
                    y=random.uniform(-40, h * 0.35),
                    vx=random.uniform(-1.2, 1.2),
                    vy=random.uniform(1.5, 4.0),
                    life=1.0,
                    color=random.choice(palette),
                    radius=2.0 + random.random() * 3.0,
                )
            )

    def flash_fade(self, amount: float = 0.55) -> None:
        self.fade = max(self.fade, amount)

    def update(self) -> None:
        alive: list[Particle] = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.12
            p.life -= 0.028
            if p.life > 0:
                alive.append(p)
        self.particles = alive

        rings_next = []
        for x, y, r, life, color in self.rings:
            life -= 0.05
            r += 3.2
            if life > 0:
                rings_next.append((x, y, r, life, color))
        self.rings = rings_next
        self.fade = max(0.0, self.fade - 0.045)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for x, y, r, life, color in self.rings:
            cv2.circle(frame, (int(x), int(y)), int(r), color, max(1, int(2 * life)), cv2.LINE_AA)

        for p in self.particles:
            a = max(0.0, min(1.0, p.life))
            col = tuple(int(c * a + 20 * (1 - a)) for c in p.color)
            cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.radius)), col, -1, cv2.LINE_AA)

        if self.fade > 0.01:
            overlay = np.full_like(frame, (18, 18, 22))
            frame = cv2.addWeighted(overlay, self.fade, frame, 1.0 - self.fade, 0)
        return frame
