import numpy as np
import random
import heapq
from collections import deque
import time
import math
from typing import List, Dict, Tuple, Set, Any
from datetime import datetime

class BayesianNetwork:
    """Enhanced Bayesian Network for uncertainty modeling"""
    def __init__(self):
        self.weather_states = ['clear', 'rain', 'storm']
        self.collapse_states = ['low', 'medium', 'high']
        self.resource_states = ['sufficient', 'limited', 'critical']
        
        # Prior probabilities
        self.priors = {
            'weather': {'clear': 0.6, 'rain': 0.3, 'storm': 0.1},
            'collapse': {'low': 0.5, 'medium': 0.3, 'high': 0.2},
            'resources': {'sufficient': 0.4, 'limited': 0.4, 'critical': 0.2}
        }
        
        # Conditional probability tables
        self.cpt = {
            'rescue|weather,collapse': {
                'clear': {
                    'low': {'success': 0.9, 'partial': 0.08, 'failure': 0.02},
                    'medium': {'success': 0.7, 'partial': 0.2, 'failure': 0.1},
                    'high': {'success': 0.4, 'partial': 0.4, 'failure': 0.2}
                },
                'rain': {
                    'low': {'success': 0.7, 'partial': 0.2, 'failure': 0.1},
                    'medium': {'success': 0.5, 'partial': 0.3, 'failure': 0.2},
                    'high': {'success': 0.3, 'partial': 0.4, 'failure': 0.3}
                },
                'storm': {
                    'low': {'success': 0.5, 'partial': 0.3, 'failure': 0.2},
                    'medium': {'success': 0.3, 'partial': 0.4, 'failure': 0.3},
                    'high': {'success': 0.1, 'partial': 0.4, 'failure': 0.5}
                }
            },
            'collapse|weather': {
                'clear': {'low': 0.7, 'medium': 0.2, 'high': 0.1},
                'rain': {'low': 0.5, 'medium': 0.3, 'high': 0.2},
                'storm': {'low': 0.3, 'medium': 0.4, 'high': 0.3}
            },
            'resources|weather,collapse': {
                'clear': {
                    'low': {'sufficient': 0.7, 'limited': 0.2, 'critical': 0.1},
                    'medium': {'sufficient': 0.5, 'limited': 0.3, 'critical': 0.2},
                    'high': {'sufficient': 0.3, 'limited': 0.4, 'critical': 0.3}
                },
                'rain': {
                    'low': {'sufficient': 0.6, 'limited': 0.3, 'critical': 0.1},
                    'medium': {'sufficient': 0.4, 'limited': 0.4, 'critical': 0.2},
                    'high': {'sufficient': 0.2, 'limited': 0.5, 'critical': 0.3}
                },
                'storm': {
                    'low': {'sufficient': 0.4, 'limited': 0.4, 'critical': 0.2},
                    'medium': {'sufficient': 0.3, 'limited': 0.4, 'critical': 0.3},
                    'high': {'sufficient': 0.1, 'limited': 0.4, 'critical': 0.5}
                }
            }
        }
        
        self.evidence_history = []
    
    def predict_rescue_success(self, evidence=None):
        """Predict rescue success probability given evidence"""
        if evidence is None:
            evidence = {}
            
        total_prob = 0
        success_prob = 0
        
        for weather in self.weather_states:
            if 'weather' in evidence and evidence['weather'] != weather:
                continue
                
            p_weather = self.priors['weather'][weather]
            
            for collapse in self.collapse_states:
                if 'collapse' in evidence and evidence['collapse'] != collapse:
                    continue
                    
                p_collapse = self.priors['collapse'][collapse]
                p_rescue = self.cpt['rescue|weather,collapse'][weather][collapse]['success']
                
                joint_prob = p_weather * p_collapse
                total_prob += joint_prob
                success_prob += joint_prob * p_rescue
        
        return success_prob / total_prob if total_prob > 0 else 0
    
    def predict_collapse_risk(self, weather_evidence):
        """Predict collapse risk given weather evidence"""
        collapse_probs = {}
        
        for collapse in self.collapse_states:
            total_prob = 0
            for weather in self.weather_states:
                if weather_evidence and weather != weather_evidence:
                    continue
                p_weather = self.priors['weather'][weather]
                p_collapse_given_weather = self.cpt['collapse|weather'][weather][collapse]
                total_prob += p_weather * p_collapse_given_weather
            
            collapse_probs[collapse] = total_prob
        
        return collapse_probs
    
    def predict_resource_status(self, evidence=None):
        """Predict resource status probability"""
        if evidence is None:
            evidence = {}
            
        resource_probs = {}
        
        for resource_state in self.resource_states:
            total_prob = 0
            
            for weather in self.weather_states:
                if 'weather' in evidence and evidence['weather'] != weather:
                    continue
                    
                p_weather = self.priors['weather'][weather]
                
                for collapse in self.collapse_states:
                    if 'collapse' in evidence and evidence['collapse'] != collapse:
                        continue
                        
                    p_collapse = self.priors['collapse'][collapse]
                    p_resources = self.cpt['resources|weather,collapse'][weather][collapse][resource_state]
                    
                    joint_prob = p_weather * p_collapse * p_resources
                    total_prob += joint_prob
            
            resource_probs[resource_state] = total_prob
        
        return resource_probs
    
    def update_with_evidence(self, new_evidence):
        """Update network with new evidence"""
        self.evidence_history.append(new_evidence)
        
        # Simple belief updating
        if 'weather' in new_evidence:
            observed_weather = new_evidence['weather']
            boost_factor = 1.2
            self.priors['weather'][observed_weather] *= boost_factor
            
            # Normalize
            total = sum(self.priors['weather'].values())
            for weather in self.weather_states:
                self.priors['weather'][weather] /= total

class AStarSearch:
    """A* Search implementation for optimal pathfinding"""
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.obstacles = set()
    
    def set_obstacles(self, obstacles):
        """Set obstacles for the grid"""
        self.obstacles = obstacles
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                (nx, ny) not in self.obstacles):
                neighbors.append((nx, ny))
        return neighbors
    
    def search(self, start, goal):
        """A* Search algorithm implementation"""
        if not (0 <= start[0] < self.grid_size and 0 <= start[1] < self.grid_size and
                0 <= goal[0] < self.grid_size and 0 <= goal[1] < self.grid_size):
            return None
            
        if start in self.obstacles or goal in self.obstacles:
            return None # Added check

        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = { (r,c): float('inf') for r in range(self.grid_size) for c in range(self.grid_size) }
        g_score[start] = 0
        f_score = { (r,c): float('inf') for r in range(self.grid_size) for c in range(self.grid_size) }
        f_score[start] = self.heuristic(start, goal)
        
        open_set_hash = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current not in open_set_hash:
                continue # Already processed
            
            open_set_hash.remove(current)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

class LogicalReasoning:
    """Logical reasoning system for victim prioritization"""
    def __init__(self):
        self.rules = [
            self.rule_injured_priority,
            self.rule_age_priority, 
            self.rule_condition_priority,
            self.rule_location_priority,
            self.rule_special_needs_priority
        ]
    
    def rule_injured_priority(self, victim):
        injured_status = victim.get('injured', False)
        if injured_status == 'unknown':
            return 10
        return 15 if injured_status else 0
    
    def rule_age_priority(self, victim):
        age = victim.get('age', 30)
        if age == 'unknown':
            return 8
        try:
            age = int(age) # Handle potential string
            if age < 12: return 12
            if age > 70: return 10
            if age < 18: return 8
            return 5
        except ValueError:
            return 8 # If still can't parse, treat as unknown
    
    def rule_condition_priority(self, victim):
        condition = victim.get('condition', 'stable')
        if condition == 'unknown':
            return 18
        return {'critical': 20, 'serious': 15, 'stable': 5}.get(condition, 0)
    
    def rule_location_priority(self, victim):
        return 10 if victim.get('in_danger_zone', False) else 0
    
    def rule_special_needs_priority(self, victim):
        return 8 if victim.get('special_needs', False) else 0
    
    def calculate_priority(self, victim):
        """Calculate priority score using all rules"""
        return sum(rule(victim) for rule in self.rules)
    
    def prioritize_victims(self, victims):
        """Sort victims by priority score - exclude rescued victims"""
        active_victims = [v for v in victims if not v.get('rescued', False)]
        if not active_victims:
            return []
            
        for victim in active_victims:
            victim['priority_score'] = self.calculate_priority(victim)
            victim['priority_factors'] = self.explain_priority(victim)
        
        return sorted(active_victims, key=lambda x: x['priority_score'], reverse=True)
    
    def explain_priority(self, victim):
        """Generate explanation for priority decision"""
        factors = []
        
        condition = victim.get('condition')
        if condition == 'critical':
            factors.append("Critical condition (+20)")
        elif condition == 'unknown':
            factors.append("Unknown condition (+18)")
        elif condition == 'serious':
            factors.append("Serious condition (+15)")

        injured = victim.get('injured')
        if injured == True:
            factors.append("Injured (+15)")
        elif injured == 'unknown':
            factors.append("Unknown injury (+10)")

        if victim.get('in_danger_zone', False):
            factors.append("In danger zone (+10)")
        if victim.get('special_needs', False):
            factors.append("Special needs (+8)")
        
        age = victim.get('age')
        if age == 'unknown':
            factors.append("Unknown age (+8)")
        else:
            try:
                age = int(age)
                if age < 12:
                    factors.append("Child (+12)")
                elif age > 70:
                    factors.append("Elderly (+10)")
                elif age < 18:
                    factors.append("Teenager (+8)")
            except ValueError:
                factors.append("Unknown age (+8)")
        
        return factors

class VolunteerAllocator:
    """Volunteer allocation system using constraint optimization"""
    def __init__(self):
        self.skills = ['medical', 'rescue', 'logistics', 'communication']
        self.zones = ['Northwest', 'Northeast', 'Southwest', 'Southeast']
    
    def optimize_allocation(self, volunteers, zone_demands, available_resources):
        """Optimize volunteer allocation based on skills and demands"""
        allocation = {zone: {skill: 0 for skill in self.skills} for zone in self.zones}
        
        sorted_volunteers = sorted(volunteers, key=lambda v: v['proficiency'], reverse=True)
        
        sorted_zones = sorted(self.zones, key=lambda z: zone_demands[z]['urgency'], reverse=True)
        
        for zone in sorted_zones:
            zone_demand = zone_demands[zone]
            
            for skill in self.skills:
                demand = zone_demand['skills'].get(skill, 0)
                available_volunteers = [v for v in sorted_volunteers 
                                        if v['skill'] == skill and v['assigned_zone'] is None]
                
                assigned_count = 0
                for volunteer in available_volunteers:
                    if assigned_count >= demand:
                        break
                    if self.can_assign_volunteer(volunteer, zone, available_resources):
                        volunteer['assigned_zone'] = zone
                        allocation[zone][skill] += 1
                        assigned_count += 1
        
        return allocation
    
    def can_assign_volunteer(self, volunteer, zone, available_resources):
        """Check if volunteer can be assigned to zone"""
        required_resources = {
            'medical': ['medical_kits', 'transport'],
            'rescue': ['equipment', 'transport'],
            'logistics': ['communication', 'vehicles'],
            'communication': ['radios', 'power']
        }
        
        skill = volunteer['skill']
        for resource in required_resources.get(skill, []):
            if available_resources[zone].get(resource, 0) <= 0:
                return False
        
        return True

## --- MODIFIED: Smarter AI for Hospital Triage ---
class HospitalTriageAI:
    """Simple rule-based AI to suggest hospital triage priority."""
    
    def __init__(self):
        # Keywords that indicate high priority
        self.critical_keywords = [
            'head', 'unconscious', 'bleeding', 'chest', 
            'breathing', 'trauma', 'crush', 'severe'
        ]

    def suggest_priority(self, report: Dict[str, Any]) -> str:
        """Suggests a priority based on condition, age, and notes."""
        
        condition = report.get('condition', 'unknown')
        age = report.get('age', 30)
        notes = report.get('notes', "").lower() # Get notes and lowercase them
        
        try:
            age = int(age)
        except ValueError:
            age = 30 # Default if unknown

        # Rule 1: Keyword check in notes
        if any(keyword in notes for keyword in self.critical_keywords):
            return 'High (Red)'

        # Rule 2: Critical condition
        if condition == 'critical':
            return 'High (Red)'
        
        # Rule 3: Serious condition + vulnerable age
        if condition == 'serious' and (age < 12 or age > 70):
            return 'High (Red)'
            
        # Rule 4: Serious condition (normal age)
        if condition == 'serious':
            return 'Medium (Yellow)'
            
        # Rule 5: Stable condition
        if condition == 'stable':
            return 'Low (Green)'
            
        # Default for 'unknown' or other cases
        return 'Medium (Yellow)'
## --- END MODIFIED ---


class DisasterScenario:
    """Disaster scenario with hospitals and zones"""
    def __init__(self, scenario_type="Earthquake", size=20):
        self.scenario_type = scenario_type
        self.size = size
        self.grid = np.zeros((size, size))
        
        self.hospitals = {
            'Hospital A': {'position': (2, 2), 'capacity': 50, 'available_beds': 35},
            'Hospital B': {'position': (17, 17), 'capacity': 75, 'available_beds': 60},
            'Hospital C': {'position': (2, 17), 'capacity': 40, 'available_beds': 25},
            'Hospital D': {'position': (17, 2), 'capacity': 60, 'available_beds': 45}
        }
        
        self.community_centers = {
            'Community Center 1': {'position': (5, 10), 'capacity': 100},
            'Community Center 2': {'position': (15, 5), 'capacity': 80},
            'Community Center 3': {'position': (10, 15), 'capacity': 120}
        }
        
        self.evacuation_shelters = {
            'North Shelter': {'position': (3, 5), 'capacity': 150},
            'South Shelter': {'position': (3, 15), 'capacity': 120},
            'East Shelter': {'position': (16, 8), 'capacity': 100},
            'West Shelter': {'position': (16, 12), 'capacity': 130}
        }
        
        self.supply_depots = {
            'Food Depot': {'position': (8, 3), 'supplies': 'Food & Water'},
            'Medical Depot': {'position': (12, 16), 'supplies': 'Medical Kits'},
            'Equipment Depot': {'position': (15, 3), 'supplies': 'Rescue Gear'}
        }
        
        self.obstacles = set()
        self.victims = []
        self.resources = []
        self.volunteers = []
        
        self.bayesian_net = BayesianNetwork()
        self.a_star_search = AStarSearch(size)
        self.logic_engine = LogicalReasoning()
        self.volunteer_allocator = VolunteerAllocator()
        
        self.triage_ai = HospitalTriageAI()
        
        self.initialize_scenario()
    
    def initialize_scenario(self):
        """Initialize scenario with realistic elements"""
        self.obstacles.clear()
        self.victims.clear()
        self.resources.clear()
        self.volunteers.clear()
        
        num_obstacles = random.randint(30, 50)
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if not self.is_position_in_facility((x, y)):
                self.obstacles.add((x, y))
                self.grid[x][y] = 1
        
        num_victims = random.randint(15, 25)
        for i in range(num_victims):
            pos = self.get_random_free_position()
            victim = {
                'id': i,
                'name': f"Victim_{i}",
                'position': pos,
                'zone': self.get_zone_name(pos),
                'injured': random.choice([True, False]),
                'age': random.randint(5, 80),
                'condition': random.choice(['critical', 'serious', 'stable']),
                'in_danger_zone': random.choice([True, False]),
                'special_needs': random.choice([True, False]),
                'rescued': False,
                'assigned_hospital': None,
                'priority_score': 0,
                'assigned_team': None, 
            }
            self.victims.append(victim)
        
        num_volunteers = random.randint(20, 30)
        skills = ['medical', 'rescue', 'logistics', 'communication']
        for i in range(num_volunteers):
            volunteer = {
                'id': i,
                'name': f"Volunteer_{i}",
                'skill': random.choice(skills),
                'proficiency': random.randint(1, 10),
                'assigned_zone': None,
                'position': self.get_random_free_position()
            }
            self.volunteers.append(volunteer)
        
        self.a_star_search.set_obstacles(self.obstacles)
    
    def is_position_in_facility(self, pos):
        """Check if position is in any facility"""
        all_facilities = list(self.hospitals.values()) + list(self.community_centers.values()) + list(self.evacuation_shelters.values()) + list(self.supply_depots.values())
        for facility in all_facilities:
            if self.get_distance(pos, facility['position']) <= 2:
                return True
        return False
    
    def get_random_free_position(self):
        """Get a random position that's not occupied"""
        max_attempts = 100
        for _ in range(max_attempts):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in self.obstacles and not self.is_position_in_facility((x, y)):
                return (x, y)
        return (5, 5)
    
    def get_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_zone_name(self, position):
        """Get the zone name based on position"""
        x, y = position
        if x < self.size/2 and y < self.size/2:
            return "Northwest"
        elif x < self.size/2 and y >= self.size/2:
            return "Southwest"
        elif x >= self.size/2 and y < self.size/2:
            return "Northeast"
        else:
            return "Southeast"
    
    def find_nearest_hospital(self, position):
        """Find the nearest hospital to a given position"""
        nearest_hospital = None
        min_distance = float('inf')
        
        for name, hospital in self.hospitals.items():
            distance = self.get_distance(position, hospital['position'])
            if distance < min_distance and hospital['available_beds'] > 0:
                min_distance = distance
                nearest_hospital = name
        
        return nearest_hospital
    
    def assign_hospital_to_victim(self, victim_position):
        """Assign the nearest available hospital to a victim"""
        hospital_name = self.find_nearest_hospital(victim_position)
        if hospital_name:
            self.hospitals[hospital_name]['available_beds'] -= 1
        return hospital_name
    
    def predict_rescue_success(self, weather, collapse_risk):
        """Predict rescue success using Bayesian network"""
        evidence = {'weather': weather, 'collapse': collapse_risk}
        return self.bayesian_net.predict_rescue_success(evidence)
    
    def predict_collapse_risk(self, weather):
        """Predict collapse risk using Bayesian network"""
        return self.bayesian_net.predict_collapse_risk(weather)
    
    def predict_resource_status(self, weather, collapse_risk):
        """Predict resource status using Bayesian network"""
        evidence = {'weather': weather, 'collapse': collapse_risk}
        return self.bayesian_net.predict_resource_status(evidence)
    
    def find_evacuation_route(self, start, goal):
        """Find route using A* algorithm"""
        return self.a_star_search.search(start, goal)
    
    def prioritize_rescue_operations(self):
        """Prioritize victims using logical reasoning - excludes rescued victims"""
        return self.logic_engine.prioritize_victims(self.victims)
    
    def get_route_to_nearest_hospital(self, start_position):
        """Get route to nearest hospital from start position"""
        hospital_name = self.find_nearest_hospital(start_position)
        if hospital_name:
            hospital_pos = self.hospitals[hospital_name]['position']
            return self.find_evacuation_route(start_position, hospital_pos), hospital_name
        return None, None
    
    def get_top_priority_victim(self):
        """Get the highest priority victim for rescue - excludes rescued victims"""
        prioritized = self.prioritize_rescue_operations()
        if prioritized:
            return prioritized[0]
        return None
    
    def get_next_sos_assignment(self):
        """Finds the highest-priority, unassigned SOS alert."""
        
        prioritized_victims = self.prioritize_rescue_operations()
        
        for victim in prioritized_victims:
            if "Citizen_SOS" in victim['name'] and victim['assigned_team'] is None:
                victim['assigned_team'] = "Assigned" 
                return victim
        
        return None
    
    def optimize_volunteer_allocation(self, zone_demands, available_resources):
        """Optimize volunteer allocation"""
        return self.volunteer_allocator.optimize_allocation(
            self.volunteers.copy(), zone_demands, available_resources
        )
    
    def get_all_destinations(self):
        """Get all possible destinations for citizens"""
        destinations = {}
        
        for name, hospital in self.hospitals.items():
            destinations[name] = {
                'type': 'hospital',
                'position': hospital['position'],
                'capacity': hospital['capacity'],
                'available_beds': hospital['available_beds'],
                'description': f"Hospital with {hospital['available_beds']} available beds"
            }
        
        for name, center in self.community_centers.items():
            destinations[name] = {
                'type': 'community_center',
                'position': center['position'],
                'capacity': center['capacity'],
                'description': f"Community Center - Capacity: {center['capacity']} people"
            }
        
        for name, shelter in self.evacuation_shelters.items():
            destinations[name] = {
                'type': 'shelter',
                'position': shelter['position'],
                'capacity': shelter['capacity'],
                'description': f"Emergency Shelter - Capacity: {center['capacity']} people"
            }
        
        for name, depot in self.supply_depots.items():
            destinations[name] = {
                'type': 'supply_depot',
                'position': depot['position'],
                'supplies': depot['supplies'],
                'description': f"Supply Depot - {depot['supplies']}"
            }
        
        return destinations

class RoleBasedAccess:
    """Role-based access control"""
    def __init__(self):
        self.role_permissions = {
            'citizen': [
                'send_sos', 'view_safe_routes', 'view_emergency_info',
                'set_location', 'view_safety_status', 'find_facility'
            ],
            'rescue_team': [
                'report_victim', 'get_assignment' 
            ],
            'emergency_coordinator': [
                'allocate_resources', 'view_analytics', 'send_alerts', 
                'optimize_deployment', 'manage_teams', 'view_system_status',
                'manage_volunteers', 'view_bayesian_analysis', 'view_sos_alerts'
            ],
            'hospital_coordinator': [
                'view_hospital_status', 'view_incoming_patients', 'triage_patients'
            ]
        }
    
    def has_permission(self, role, action):
        return action in self.role_permissions.get(role, [])
    
    def get_available_actions(self, role):
        return self.role_permissions.get(role, [])