# S26-41 Capstone Project

All code files associated with the Capstone team S26-41.

## Project Overview

This is an EEG-based biofeedback relaxation system that streams real-time brainwave data, measures relaxation state, and provides audio-visual feedback to help users achieve flow states through adaptive audio volume control.

## Tech Stack

- **Language**: Python
- **EEG Hardware**: OpenBCI Cyton + Daisy
- **GUI Framework**: PyQt5
- **EEG Processing**: BrainFlow
- **Visualization**: Matplotlib

## Getting Started

### Prerequisites

- Python 3.7 or higher
- OpenBCI Cyton + Daisy EEG headset
- Linux system with ALSA audio support (for amixer)
- mpv media player (for video playback)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KeenShield/S26-41-Code-Repo.git
   cd S26-41-Code-Repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure & File Descriptions

### UI Files

- **ActualUI.py**: Main session interface UI
  - Displays relaxation metric with progress bar
  - Shows time elapsed in MM:SS format
  - Dynamic color indicator for relaxation state
  - Control buttons: Play, Stop, Cycle audio, EXIT session
  - System volume control bar and active EEG channel display
  - Placeholder frames for EEG power spectral density (PSD) plotting

- **ExampleUI.py**: Demonstration/test UI with minimal components
  - Shows how to set up a PyQt5 window with matplotlib integration
  - Includes a simple checkbox and button to trigger a sine wave plot
  - Used as a reference or testing template for UI development

- **SessionStats.py**: Post-session statistics display screen
  - Shows session duration, time spent in relaxed state
  - Displays average relaxation score and most "relaxing" audio track
  - Provides buttons to: Close, Redo Session, and Download results
  - Contains frame placeholders for relaxation vs time and audio vs time graphs

- **tempUIStats.py**: Alternative/temporary session statistics UI
  - Similar to SessionStats.py with minor layout differences
  - Shows session duration, time relaxed, and average relaxation score
  - Displays "Most Engaging Audio"
  - Contains Close and Redo Session buttons

- **tempUIplacer.py**: Duplicate/temporary version of the main session UI
  - Nearly identical to ActualUI.py
  - May be used for testing or development purposes

- **Flow.ui**: Qt Designer XML layout file for the main session interface
  - Contains visual layout specifications for ActualUI.py and tempUIplacer.py
  - Defines widget positioning, sizes, colors, and styling
  - Auto-generates Python code via PyQt5's pyuic5 tool

### Core Functionality Files

- **relaxation_metric.py**: Main project file - Computes relaxation state from EEG frequency bands
  - Analyzes EEG power in alpha, theta, and other bands to determine relaxation level
  - Provides baseline calibration and z-score normalization
  - Contains mathematical formulas for relaxation detection
  - Integrates GUI updates with real-time EEG processing
  - Manages video playback and audio selection logic

- **focus_stream.py**: EEG data streaming and processing engine
  - Connects to OpenBCI Cyton + Daisy EEG headset via BrainFlow library
  - Streams raw EEG data and computes frequency band powers (delta, theta, alpha, beta, gamma)
  - Calculates relaxation and focus metrics from EEG band powers
  - Auto-detects active EEG channels based on signal variance
  - Performs optional baseline calibration for z-score normalization
  - Maps relaxation metric to audio volume in real-time
  - Outputs debug metrics and per-channel analysis

- **volume_control.py**: Audio volume mapping and playback control
  - `map_metric_to_volume()`: Converts relaxation/focus metrics to 0-1 volume scale with optional inversion
  - `set_system_volume_percent()`: Sets system ALSA Master volume using amixer
  - `TonePlayer`: Background sine wave tone generator with real-time amplitude control
  - Supports linear and exponential volume curves for smooth transitions

### Testing/Demo Files

- **tone_test.py**: Video playback test with real-time volume control
  - Plays an MP4 video (trippy3.mp4) full-screen using mpv
  - Demonstrates live system volume changes via amixer
  - Tests volume mapping with demo gain values (10%, 80%, 20%, 100%, 35%, 60%)
  - Useful for testing audio-visual feedback during sessions

### Configuration Files

- **requirements.txt**: Project dependencies list
  - Lists all Python packages needed to run the project

## Usage

Run the main application:
```bash
python relaxation_metric.py
```

This will start the GUI and begin EEG streaming from your connected Cyton + Daisy headset on `/dev/ttyUSB0`.

For testing video playback and volume control:
```bash
python tone_test.py
```

For streaming EEG data and computing metrics (without GUI):
```bash
python focus_stream.py --port /dev/ttyUSB0 --window 2.0 --interval 1.0 --calibrate 5.0
```

## Contributing

Team S26-41 members should follow the project's contribution guidelines.

## License

[Specify license if applicable]

## Contact

For questions or inquiries, contact the S26-41 Capstone team.