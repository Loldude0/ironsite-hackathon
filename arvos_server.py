import asyncio
from arvos import ArvosServer

async def main():
    server = ArvosServer(port=9090)

    server.print_qr_code()
    print("Waiting for iPhone connection...")

    imu_seen = False

    async def on_connect(client):
        print("iPhone connected ✅")

    async def on_imu(data):
        nonlocal imu_seen
        if not imu_seen:
            print("IMU streaming started 📡")
            imu_seen = True

        print(
            f"accel={data.linear_acceleration} | "
            f"gyro={data.angular_velocity}"
        )

    server.on_connect = on_connect
    server.on_imu = on_imu

    await server.start()

asyncio.run(main())