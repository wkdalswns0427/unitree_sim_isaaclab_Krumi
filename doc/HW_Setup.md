**1. System and Hardware Requirements**
*   **Operating System:** Ubuntu 20.04 is highly recommended for development. Mac and Windows systems are currently not supported .
*   **Onboard Computers:** The H1-2 comes equipped with a motion control unit (PC1, IP: `192.168.123.161`), which is strictly reserved for Unitree's internal motion control and is not open to users. For secondary development, you must use the onboard development unit PC4 (IP: `192.168.123.164`) .

**2. Network Configuration**
*   Connect your development computer to the H1-2 switch using an Ethernet cable .
*   Configure your computer's wired network adapter to be on the `192.168.123.x` subnet (e.g., set the IP address to `192.168.123.222` or `192.168.123.99`) .
*   Verify the connection by executing `ping 192.168.123.161` in your terminal. A successful reply indicates the network is configured correctly .
*   Use the `ifconfig` command to identify the exact name of your network interface card (NIC) associated with the `123` subnet (for example, `enxf8e43b808e06`). You will need to provide this interface name as a parameter when running the examples .

**3. SDK Installation and Compilation**
*   Download the official `unitree_sdk2` from GitHub: `https://github.com/unitreerobotics/unitree_sdk2` .
*   Navigate into the `unitree_sdk2` directory, and compile the SDK by running the following commands:
    ```bash
    mkdir build
    cd build
    cmake ..
    sudo make install
    ```
*   You can compile the provided examples (such as high-level or low-level routines) by running the `make` command inside their respective directories. The generated binary executables will be located in the `build` folders .

**4. Entering Debug Mode and Running Examples**
*   **Debug Mode (Crucial for Low-Level Control):** If you are running low-level development examples, the built-in motion control program will conflict with your SDK commands and cause severe jittering. To prevent this, you **must** enter Debug Mode . 
    1. Suspend the robot securely using a protective hoist .
    2. Ensure the robot is in damping mode, then press the **L2 + R2** buttons simultaneously on the remote control to enter Debug Mode .
    3. You can press **L2 + A** to verify; the robot should strike a specific diagnostic posture to indicate it has successfully entered Debug Mode .
*   Run the compiled example program by specifying your NIC name:
    ```bash
    sudo ./<example_executable> <your_NIC_name>
    ```
    *(e.g., `sudo ./h1_low_level_example enxf8e43b808e06`)* .

**5. Untethered Deployment (Optional)**
If you wish to test without the Ethernet cable tethered to your user PC, you can deploy your application directly onto the H1-2's development unit (PC4):
*   Connect to PC4 via SSH: `ssh unitree@192.168.123.164` (Contact technical support for the initial password if necessary) .
*   It is recommended to compile the executable directly on PC4 to avoid dependency issues. Run the program in the background via SSH, and once confirmed it is running normally, you can disconnect the Ethernet cable for untethered testing .