from matplotlib import pyplot as plt
from absl import flags, app
import numpy as np



def main(argv):
    del argv
    
    num_clients = 10
    bs_icon = "./assets/icons/cell-tower1.png"
    iot_icon = "./assets/icons/phone1.png"
    # Simulate the locations of the clients
    radius = 100  # radius of the circle
    angles = np.random.uniform(0, 2 * np.pi, num_clients)
    distances = np.random.normal(radius / 2, radius / 4, num_clients)

    # Ensure distances are within the circle
    distances = np.clip(distances, 0, radius)

    # Calculate the x and y coordinates of the clients
    x_coords = distances * np.cos(angles)
    y_coords = distances * np.sin(angles)
    
    # plot the bs and clients
    plt.figure(figsize=(5, 5))
    bs_icon_img = plt.imread(bs_icon)
    
    icon_size = 0.5
    # plt.scatter(0, 0, c='red', label='BS')
    plt.imshow(bs_icon_img, extent=(-4, 4, -10, 10), aspect='auto', zorder=5)
    plt.scatter(x_coords, y_coords, c='blue', label='Clients')
    # icon_img = plt.imread(iot_icon)
    # for x, y in zip(x_coords, y_coords):
    #     plt.imshow(icon_img, extent=(x - icon_size / 2, x + icon_size / 2, 
    #                                  y - icon_size / 2, y + icon_size / 2),
    #                                    aspect='auto', zorder=4)
    # plt.xlim(-radius, radius)
    # plt.ylim(-radius, radius)
    plt.legend()
    plt.grid()
    plt.savefig('./assets/icons/ue.png', dpi=300)

if __name__ == '__main__':
    app.run(main)   