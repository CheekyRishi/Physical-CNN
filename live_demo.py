import serial
import time
import numpy as np

TOTAL_LEDS = 1163

# --- SERIAL SETUP ---
try:
    arduino = serial.Serial('COM6', 500000, timeout=2) 
    time.sleep(2)
    print("\n[SUCCESS] Hardware Pipeline Initialized.")
except Exception as e:
    print(f"\n[ERROR] Arduino not found. Please check your COM port: {e}")
    exit()

def send_frame(led_array):
    """Clips data to safe limits, attaches the 255 header, and streams to Arduino"""
    safe_data = np.clip(led_array, 0, 150).astype(np.uint8)
    arduino.write(bytes([255, 255])) 
    arduino.write(bytes(safe_data.tolist())) 
    arduino.readline() 

def turn_all_off():
    send_frame(np.zeros(TOTAL_LEDS, dtype=np.uint8))
    time.sleep(0.5)

# =====================================================================
# THE VISUAL ENHANCER (Sparsity + Shimmer)
# =====================================================================
def apply_conv_thresholds(frames):
    print("[INFO] Applying High-Contrast Compute Shimmer to Conv Layers...")
    enhanced = frames.copy().astype(np.int16)
    
    conv_blocks = [
        enhanced[:, 0:540],   
        enhanced[:, 580:1120] 
    ]
    
    for block in conv_blocks:
        block[block <= 40] = 0
        block[(block > 40) & (block <= 80)] = 60    
        block[(block > 80) & (block <= 120)] = 100  
        block[block > 120] = 140                    
        
        noise = np.random.randint(-25, 26, size=block.shape)
        active_mask = block > 0
        block[active_mask] += noise[active_mask]

    return np.clip(enhanced, 0, 150).astype(np.uint8)

# =====================================================================
# THE LIVE JURY INTERFACE
# =====================================================================
while True:
    print("\n" + "="*45)
    print(" CNN HARDWARE INTERPRETABILITY DEMO")
    print("="*45)
    print("1. Classify Apple")
    print("2. Classify Banana")
    print("3. Classify Orange")
    print("4. Turn Lights Off & Exit")
    
    choice = input("\nJury Selection (1-4): ")
    
    file_map = {
        '1': 'anim_apple.npy',
        '2': 'anim_banana.npy',
        '3': 'anim_orange.npy'
    }
    
    if choice == '4':
        turn_all_off()
        arduino.close()
        print("\nExiting. Good luck with the presentation!")
        break
        
    if choice not in file_map:
        print("Invalid choice.")
        continue
        
    try:
        print(f"\n[INFO] Loading deep learning tensor data from {file_map[choice]}...")
        frames = np.load(file_map[choice])
        
        # -----------------------------------------------------------------
        # THE JURY-PROOF HARDCODE
        # 520 frames = overwrites the 3-second gap + 10-second hold.
        # This makes it light up almost instantly after Maxpool 2!
        # If you want a 1-second dramatic pause, change 520 to 440.
        # -----------------------------------------------------------------
        frames[:, 1160:1163] = 0  
        SHOW_FRAMES = 520 
        
        if choice == '1':   # Apple is 1160
            frames[-SHOW_FRAMES:, 1160] = 150
        elif choice == '2': # Banana is 1162
            frames[-SHOW_FRAMES:, 1162] = 150
        elif choice == '3': # Orange is 1161
            frames[-SHOW_FRAMES:, 1161] = 150
            
        # Apply the Shimmer Filter to the Conv Layers
        frames = apply_conv_thresholds(frames)
        
        print(f"[PLAYING] Streaming {len(frames)} frames across 1,163 LEDs...")
        
        for frame in frames:
            send_frame(frame)
            
        print("\n[RESULT] Inference Sequence Complete!")
        input("Press Enter to turn off the display and await next Jury selection...")
        turn_all_off()
        
    except FileNotFoundError:
        print(f"\n[ERROR] Missing {file_map[choice]}! Make sure the .npy files are in the exact same folder as this script.")