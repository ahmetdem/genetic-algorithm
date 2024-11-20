# Import required libraries
from PIL import Image
import numpy as np
import random, math
import matplotlib.pyplot as plt

def generate_population(image_path, num_patches, population_size):
    """
    Generate a population of shuffled images for the genetic algorithm.

    Parameters:
        image_path (str): Path to the input image.
        num_patches (int): Desired number of patches.
        population_size (int): Number of individuals in the population.

    Returns:
        population (list): List of individuals, each represented by shuffled indices.
        patch_size (int): Calculated patch size for the given number of patches.
    """

    # Load the image, resize to 128x64 pixels, and convert to grayscale
    img = Image.open(image_path).resize((128, 64)).convert("L")  # Resize and grayscale
    img = np.array(img)

    # Get image dimensions
    height, width = img.shape

    # Calculate patch size
    total_pixels = height * width
    patch_area = total_pixels / num_patches
    patch_size = int(math.sqrt(patch_area))

    # Ensure patch size divides the image dimensions perfectly
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Specified number of patches does not allow exact tiling of the image. Try a different value.")

    # Break the image into patches
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    # Generate population
    population = []
    for _ in range(population_size):
        shuffled_indices = list(range(len(patches))) 
        random.shuffle(shuffled_indices)
        population.append(shuffled_indices)  

    return population, patch_size, patches, img

image_path = "img.png" 
num_patches = 8 
population_size = 20

population, patch_size, patches, original_img = generate_population(image_path, num_patches, population_size)

print("Patch size:", patch_size)
print("Population size:", population_size)
print("Number of patches:", num_patches)
print("Number of patches in the image:", len(patches))
print("Population length:", len(population))
print("Population example:", population[0])

# Display an individual from the population (for visualization)
shuffled_indices_example = population[0]  # Example individual
shuffled_patches_example = [patches[i] for i in shuffled_indices_example]

# Reconstruct the shuffled image for the example individual
height, width = original_img.shape
shuffled_img_example = np.zeros_like(original_img)
idx = 0
for i in range(0, height, patch_size):
    for j in range(0, width, patch_size):
        shuffled_img_example[i:i+patch_size, j:j+patch_size] = shuffled_patches_example[idx]
        idx += 1
        
# Visualize the example individual's shuffled image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Shuffled Image (Example Individual)")
plt.imshow(shuffled_img_example, cmap="gray")
plt.axis("off")
plt.show()

def calculate_fitness(individual, original_indices):
    """
    Calculate the fitness of an individual based on patch correctness.
    
    Parameters:
        individual (list): The shuffled indices of the individual's patches.
        original_indices (list): The correct indices of the patches.
    
    Returns:
        fitness (int): Number of patches in the correct position.
    """
    # Count the number of patches that are in the correct position
    fitness = sum(1 for i, idx in enumerate(individual) if idx == original_indices[i])
    return fitness
# Calculate the fitness of all of the population members and show it like a graph with matplotlib
original_indices = list(range(len(patches)))  # Original order of patches
fitness_scores = [calculate_fitness(individual, original_indices) for individual in population]

plt.figure(figsize=(10, 5))
plt.plot(range(1, population_size + 1), fitness_scores, marker="o", linestyle="-")
plt.title("Fitness Scores of the Population")
plt.xlabel("Individual")
plt.ylabel("Fitness Score")
plt.grid(True)
plt.show(),

def order_crossover(parent1, parent2):
	"""
	Perform Order Crossover (OX) on two parents.

	Parameters:
		parent1 (list): First parent individual.
		parent2 (list): Second parent individual.

	Returns:
		child1, child2 (list): Two offspring individuals.
	"""
	size = len(parent1)
	start, end = sorted(random.sample(range(size), 2))  # Select two crossover points


	# Create children with placeholder values
	child1 = [-1] * size
	child2 = [-1] * size

	# Copy the segment from parent1 to child1 and parent2 to child2
	child1[start:end] = parent1[start:end]
	child2[start:end] = parent2[start:end]

	# Fill the remaining slots with the order of genes from the other parent
	def fill_child(child, parent):
		current_pos = end
		for gene in parent:
			if gene not in child:
				if current_pos >= size:
					current_pos = 0
				child[current_pos] = gene
				current_pos += 1

	fill_child(child1, parent2)
	fill_child(child2, parent1)

	return child1, child2

def swap_mutation(individual, mutation_rate):
    """
    Perform swap mutation on an individual.

    Parameters:
        individual (list): Individual to mutate.
        mutation_rate (float): Probability of mutation for each gene.

    Returns:
        mutated (list): Mutated individual.
    """
    mutated = individual[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]  # Swap genes
    return mutated

def tournament_selection(population, fitness_scores, k=3):
    """
    Perform tournament selection.

    Parameters:
        population (list): Current population of individuals.
        fitness_scores (list): List of fitness scores for each individual.
        k (int): Number of individuals to participate in each tournament.

    Returns:
        selected (list): A list of selected individuals for the next generation.
    """

    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), k)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def evolve_population(population, fitness_scores, crossover_rate, mutation_rate, elitism_size=2):
    """
    Evolve the population by performing selection, crossover, and mutation.

    Parameters:
        population (list): Current population of individuals.
        fitness_scores (list): List of fitness scores for each individual.
        crossover_rate (float): Probability of crossover between two individuals.
        mutation_rate (float): Probability of mutation for each individual.
        elitism_size (int): Number of top individuals to carry over to the next generation.

    Returns:
        new_population (list): Population of individuals evolved.
    """

    # Step 1: Selection with tournament selection
    selected_population = tournament_selection(population, fitness_scores)

    # Step 2: Elitism - carry over the best individuals
    elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elitism_size]
    new_population = [population[i] for i in elite_indices]
    
    # Step 3: Crossover
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(selected_population, 2)
        if random.random() < crossover_rate:
            child1, child2 = order_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
            
        new_population.extend([child1, child2])
    
    # Step 4: Mutation
    new_population = [swap_mutation(individual, mutation_rate) for individual in new_population[:len(population)]]
    
    return new_population

def visualize_population(population, patches, patch_size, generation, fitness_scores):
    """
    Visualizes the image based on the current population and their fitness.

    Parameters:
        population (list): Current population of individuals.
        patches (list): List of image patches.
        patch_size (int): Size of each patch.
        generation (int): Current generation for naming.
    """
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    best_image = np.zeros_like(original_img)

    # Reconstruct the image from patches
    idx = 0
    for i in range(0, len(best_image), patch_size):
        for j in range(0, len(best_image[0]), patch_size):
            patch = patches[best_individual[idx]]
            best_image[i:i+patch_size, j:j+patch_size] = patch
            idx += 1

    # Display the image
    plt.imshow(best_image, cmap='gray')
    plt.title(f"Generation {generation + 1}")
    plt.show()
    
population_size = 20
num_generations = 1000
crossover_rate = 0.8
mutation_rate = 0.1

# Evolve the population
for generation in range(num_generations):
    # Calculate fitness scores for the population
    fitness_scores = [calculate_fitness(ind, list(range(num_patches))) for ind in population]

    # Print best fitness in the current generation
    best_fitness = max(fitness_scores)
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    if best_fitness == num_patches:
        print(f"Solution found in Generation {generation + 1}")
        visualize_population(population, patches, patch_size, generation, fitness_scores)
        break

    # Uncooment the line below to visualize the individuals in each generation
    # visualize_population(population, patches, patch_size, generation, fitness_scores)

    # Evolve the population
    population = evolve_population(population, fitness_scores, crossover_rate, mutation_rate)

# Evolve the population
generations_to_solution = []

for _ in range(100):
    population, patch_size, patches, original_img = generate_population(image_path, num_patches, population_size)
    for generation in range(num_generations):
        # Calculate fitness scores for the population
        fitness_scores = [calculate_fitness(ind, list(range(num_patches))) for ind in population]

        # Print best fitness in the current generation
        best_fitness = max(fitness_scores)
        # print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        if best_fitness == num_patches:
            generations_to_solution.append(generation + 1)
            break

        # Evolve the population
        population = evolve_population(population, fitness_scores, crossover_rate, mutation_rate)

average_generations = sum(generations_to_solution) / len(generations_to_solution)
print(f"Average generations to find solution: {average_generations}")