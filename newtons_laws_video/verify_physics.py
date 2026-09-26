"""Independent check of every number quoted in the lesson narration.

Run:  python verify_physics.py
Each check recomputes a quoted value from first principles and asserts that the
value spoken in the video matches it to the precision it is quoted with.
"""
import math

g = 9.8  # m/s^2, the value used throughout the video
checks = []


def check(label, computed, quoted, tol):
    ok = abs(computed - quoted) <= tol
    checks.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {label}: computed {computed:.6g}, quoted {quoted}")


# --- Chapter 1: friction stopping distances (v0 = 2 m/s), d = v0^2 / (2 mu g)
for mu, d_q in [(0.3, 0.68), (0.1, 2.04), (0.02, 10.2)]:
    check(f"stopping distance mu={mu}", 2.0**2 / (2 * mu * g), d_q, 0.005)

# --- Chapter 3: momentum table, p = m v
check("bicycle + rider 80 kg at 5 m/s", 80 * 5, 400, 1e-9)
check("loaded truck 20 000 kg at 5 m/s", 20000 * 5, 100000, 1e-9)
check("slow cricket ball 0.16 kg at 10 m/s", 0.16 * 10, 1.6, 1e-9)
check("fast cricket ball 0.16 kg at 40 m/s", 0.16 * 40, 6.4, 1e-9)

# --- Chapter 3: carts
check("1 kg cart, 1 N -> a", 1 / 1, 1, 1e-9)
check("1 kg cart, 2 N -> a", 2 / 1, 2, 1e-9)
check("2 kg cart, 2 N -> a", 2 / 2, 1, 1e-9)
check("distance in 2 s at a=2", 0.5 * 2 * 2**2, 4, 1e-9)
check("distance in 2 s at a=1", 0.5 * 1 * 2**2, 2, 1e-9)

# --- Chapter 3: projectile, v0 = (4, 6) m/s
T_flight = 2 * 6 / g
check("projectile flight time (s)", T_flight, 1.224, 0.001)
check("projectile range (m)", 4 * T_flight, 4.9, 0.01)
check("projectile max height (m)", 6**2 / (2 * g), 1.84, 0.005)

# --- Chapter 3: force switched off after 2 s (1 kg, 1 N)
check("velocity when the force stops (m/s)", 1 / 1 * 2, 2, 1e-9)
check("distance while the force acts (m)", 0.5 * 1 * 2**2, 2, 1e-9)
check("distance after 4 s (m)", 2 + 2 * 2, 6, 1e-9)

# --- Chapter 4: impulse, cricket ball 0.16 kg at 25 m/s brought to rest
dp = 0.16 * 25
check("cricket ball momentum change (kg m/s)", dp, 4, 1e-9)
check("average force, stopped in 0.01 s (N)", dp / 0.01, 400, 1e-6)
check("average force, stopped in 0.1 s (N)", dp / 0.1, 40, 1e-6)

# --- Chapter 5: skaters (Anna 50 kg, Ben 75 kg), 750 N push for 0.2 s
F, t_push = 750, 0.2
check("Anna acceleration", F / 50, 15, 1e-9)
check("Ben acceleration", F / 75, 10, 1e-9)
check("Anna final speed", F / 50 * t_push, 3, 1e-9)
check("Ben final speed", F / 75 * t_push, 2, 1e-9)
check("separation during the push (m), within arm reach", 0.5 * (F / 50 + F / 75) * t_push**2, 0.5, 1e-9)

# --- Chapter 5: apple and Earth
m_apple, M_earth = 0.1, 5.97e24
F_apple = m_apple * g
check("gravitational force on 0.1 kg apple (N)", F_apple, 0.98, 1e-9)
check("Earth's acceleration (x1e-25 m/s^2)", F_apple / M_earth / 1e-25, 1.6, 0.05)
check("Earth's acceleration with M = 6.0e24 (x1e-25)", F_apple / 6.0e24 / 1e-25, 1.6, 0.05)
check("apple acceleration", F_apple / m_apple, 9.8, 1e-9)

# --- Chapter 6: momentum after the push
check("Anna momentum", 50 * -3, -150, 1e-9)
check("Ben momentum", 75 * 2, 150, 1e-9)
check("total momentum", 50 * -3 + 75 * 2, 0, 1e-9)

# --- Chapter 7, example 1: 5 kg box, 20 N at 30 deg, frictionless floor
m, P, th = 5, 20, math.radians(30)
check("weight of 5 kg box (N)", m * g, 49, 1e-9)
check("horizontal component (N)", P * math.cos(th), 17.3, 0.05)
check("vertical component (N)", P * math.sin(th), 10, 1e-9)
check("acceleration (m/s^2)", P * math.cos(th) / m, 3.46, 0.005)
check("normal force (N)", m * g - P * math.sin(th), 39, 1e-9)

# --- Chapter 7, example 2: 60 kg student in a lift, N = m (g + a)
m = 60
check("constant velocity: N", m * (g + 0), 588, 1e-9)
check("accelerating up 2: N", m * (g + 2), 708, 1e-9)
check("accelerating down 2: N", m * (g - 2), 468, 1e-9)
check("free fall: N", m * (g - g), 0, 1e-9)
check("g + a", g + 2, 11.8, 1e-9)
check("g - a", g - 2, 7.8, 1e-9)

# --- Chapter 7, example 3: 4 kg (front) + 2 kg (rear), 12 N pull on the front block
a = 12 / (4 + 2)
T = 2 * a
check("system acceleration", a, 2, 1e-9)
check("string tension", T, 4, 1e-9)
check("front block: 12 - T = 4a", 12 - T, 4 * a, 1e-9)

# --- Chapter 8, quiz: Atwood machine 3 kg and 5 kg
m1, m2 = 3, 5
a = (m2 - m1) * g / (m1 + m2)
T = 2 * m1 * m2 * g / (m1 + m2)
check("Atwood acceleration", a, 2.45, 1e-9)
check("Atwood tension", T, 36.75, 1e-9)
check("T = m1 (g + a)", m1 * (g + a), 36.75, 1e-9)
check("T = m2 (g - a)", m2 * (g - a), 36.75, 1e-9)
check("3 kg weight", m1 * g, 29.4, 1e-9)
check("5 kg weight", m2 * g, 49, 1e-9)

# --- Galileo ramps: equal release/return height for any second-ramp angle (energy)
h = 2.0
for deg in (35, 20, 12):
    v_bottom = math.sqrt(2 * (5 / 7) * g * math.sin(math.radians(35)) * h / math.sin(math.radians(35)))
    # distance up the second ramp with rolling deceleration (5/7) g sin(theta)
    d_up = v_bottom**2 / (2 * (5 / 7) * g * math.sin(math.radians(deg)))
    check(f"height regained on {deg} deg ramp", d_up * math.sin(math.radians(deg)), h, 1e-9)

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
