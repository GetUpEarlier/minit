from .library import launch_server
import base64

def main():
    id = launch_server()
    print(f"server launched at: {id}")
    base64_id = base64.b64encode(id)
    print(f"base64: {base64_id}")
    print("Press enter to exit")
    input()


if __name__ == '__main__':
    main()