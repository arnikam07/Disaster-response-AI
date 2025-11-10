import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any

# --- Import all classes from your backend file ---
from backend import *

# Set page configuration
st.set_page_config(
    page_title="Disaster Response AI Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DisasterResponseAI:
    """Main application class"""
    def __init__(self):
        self.access_control = RoleBasedAccess()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        default_states = {
            'scenario': None,
            'alerts': [],
            'rescue_requests': [],
            'hospital_messages': [], 
            'user_location': None,
            'user_zone': None,
            'user_role': 'citizen',
            'weather_condition': 'clear',
            'collapse_risk': 'medium',
            'selected_destination': None,
            'destination_route': None,
            'sos_sent_message': None, 
            'team_location': None,      
            'current_assignment': None, 
            'reporting_victim': False, 
            'resource_status': {
                'medical_kits': 100, 
                'food_supplies': 200, 
                'water': 150, 
                'equipment': 50, 
                'transport': 30,
                'communication': 25,
                'power': 40
            },
            'volunteer_allocation': None,
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        if st.session_state.scenario is None:
            st.session_state.scenario = DisasterScenario()

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_content()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("Disaster Response AI")
            
            st.subheader("User Role")
            role = st.radio(
                "Select your role:",
                ["Citizen", "Rescue Team", "Emergency Coordinator", "Hospital Coordinator"],
                key="role_selector"
            )
            st.session_state.user_role = role.lower().replace(" ", "_")
            
            st.subheader("Scenario Configuration")
            scenario_type = st.selectbox(
                "Disaster Type:",
                ["Earthquake", "Flood", "Hurricane", "Wildfire", "Tornado"]
            )
            
            st.subheader("Environment Conditions")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.weather_condition = st.selectbox(
                    "Weather:",
                    ["clear", "rain", "storm"]
                )
            with col2:
                st.session_state.collapse_risk = st.selectbox(
                    "Collapse Risk:",
                    ["low", "medium", "high"]
                )
            
            if st.button("Initialize Scenario", use_container_width=True):
                self.initialize_session_state() 
                st.session_state.scenario = DisasterScenario(scenario_type=scenario_type) 
                st.session_state.user_role = role.lower().replace(" ", "_") 
                st.success("Scenario initialized!")
                st.rerun()
            
            st.divider()
            
            st.subheader("Your Location")
            if st.session_state.user_location:
                st.write(f"Zone: {st.session_state.user_zone}")
                st.write(f"Coordinates: {st.session_state.user_location}")
            
            if st.session_state.team_location and st.session_state.user_role == 'rescue_team':
                st.subheader("Team Status")
                st.write(f"Team Location: {st.session_state.team_location}")
            
            st.divider()
            st.subheader("Active Alerts")
            for alert in st.session_state.alerts[-3:]:
                st.error(f"⚠ {alert}")

    def render_main_content(self):
        """Render main content"""
        st.title("Emergency Response & Hospital Routing System")
        
        role_display = st.session_state.user_role.replace("_", " ").title()
        st.header(f"Welcome, {role_display}")
        
        self.render_quick_actions()
        
        if st.session_state.user_role == 'hospital_coordinator':
            tabs = st.tabs(["Dashboard", "Emergency Map", "Hospitals & Zones"])
        else:
            tabs = st.tabs(["Dashboard", "Emergency Map", "Hospitals & Zones", "AI Coordination"])
        
        with tabs[0]:
            self.render_dashboard()
        with tabs[1]:
            self.render_emergency_map()
        with tabs[2]:
            self.render_hospitals_zones()
        
        if st.session_state.user_role != 'hospital_coordinator':
            with tabs[3]:
                self.render_ai_coordination()

    def render_quick_actions(self):
        """Render quick actions in main dashboard"""
        st.subheader("Quick Actions")
        
        role = st.session_state.user_role
        
        if role == "citizen":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Set Location", use_container_width=True):
                    self.set_user_location()
            with col2:
                if st.button("Send SOS Alert", use_container_width=True):
                    self.send_sos_alert()
                    
        elif role == "rescue_team":
            if st.session_state.team_location is None:
                if st.button("Set Team Start Location", use_container_width=True, type="primary"):
                    st.session_state.team_location = st.session_state.scenario.get_random_free_position()
                    st.rerun()
            else:
                st.info(f"Your team is active. See dashboard for assignments.")
                    
        elif role == "emergency_coordinator":
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("System Status", use_container_width=True):
                    self.show_system_status()
            with col2:
                if st.button("Allocate Volunteers", use_container_width=True):
                    self.allocate_volunteers()
            with col3:
                if st.button("Send Alert", use_container_width=True):
                    self.send_emergency_alert()
        
        elif role == "hospital_coordinator":
            st.info("Your role is to monitor incoming patients and hospital capacity.")

        st.divider()

    def render_alert_box(self):
        """Displays the central alert box for ALL alerts (for non-citizens)."""
        if st.session_state.alerts:
            st.warning("Incoming Alerts:")
            for alert in st.session_state.alerts[-3:]: # Show top 3
                st.error(f"⚠ {alert}")

    ## --- NEW: Function for citizens to see ONLY official alerts ---
    def render_broadcast_alerts(self):
        """Displays only OFFICIAL broadcast alerts for citizens."""
        # Filter for official alerts
        official_alerts = [a for a in st.session_state.alerts if a.startswith("OFFICIAL ALERT:")]
        
        if official_alerts:
            st.warning("Official Broadcasts:")
            for alert in official_alerts[-3:]: # Show top 3
                # Clean up the message
                alert_message = alert.replace("OFFICIAL ALERT: ", "")
                st.error(f"⚠ {alert_message}")
    ## --- END NEW ---

    def render_dashboard(self):
        """Render role-specific dashboard"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.user_role == "citizen":
                self.render_citizen_dashboard()
            elif st.session_state.user_role == "rescue_team":
                self.render_rescue_team_dashboard()
            elif st.session_state.user_role == "emergency_coordinator":
                self.render_coordinator_dashboard()
            elif st.session_state.user_role == "hospital_coordinator":
                self.render_hospital_coordinator_dashboard()
        
        with col2:
            self.render_quick_status()

    def render_citizen_dashboard(self):
        """Render dashboard for citizens"""
        
        ## --- MODIFIED: Replaced general alert box with specific one ---
        self.render_broadcast_alerts()
        ## --- END MODIFIED ---
        
        if st.session_state.sos_sent_message:
            st.success(st.session_state.sos_sent_message)
        
        st.subheader("Your Safety Status")
        
        safety_status = self.calculate_safety_status()
        if safety_status == "safe":
            st.success("You are in a safe location")
        elif safety_status == "warning":
            st.warning("Take caution - moderate risk area")
        else:
            st.error("HIGH RISK - Evacuate immediately!")
        
        if st.session_state.user_location:
            st.write(f"*Your Location:* {st.session_state.user_zone}")
            st.write(f"*Coordinates:* {st.session_state.user_location}")
        
        st.divider()

        st.subheader("Find Nearest Facility")
        if not st.session_state.user_location:
            st.warning("Set your location using the 'Quick Action' button above to find routes.")
        else:
            dest_types = {
                "Hospitals": 'hospital',
                "Shelters": 'shelter',
                "Community Centers": 'community_center',
                "Supply Depots": 'supply_depot'
            }
            selected_type_name = st.selectbox("Select facility type:", dest_types.keys())
            selected_type_key = dest_types[selected_type_name]
            
            all_dests = st.session_state.scenario.get_all_destinations()
            
            relevant_dests_raw = {k: v for k, v in all_dests.items() if v['type'] == selected_type_key}
            relevant_dests = {}

            for name, data in relevant_dests_raw.items():
                if selected_type_key == 'hospital':
                    if data['available_beds'] > 0:
                        relevant_dests[name] = data
                else:
                    relevant_dests[name] = data
            
            if not relevant_dests:
                st.error(f"No available {selected_type_name} found in the scenario.")
            else:
                dest_name = st.selectbox(f"Choose {selected_type_name}:", relevant_dests.keys())
                
                if st.button(f"Find Route to {dest_name}"):
                    self.calculate_destination_route(dest_name, relevant_dests[dest_name])

        if st.session_state.destination_route:
            st.success(f"Route calculated to {st.session_state.selected_destination}! View on 'Emergency Map' tab.")
            st.write(f"*Route Steps:* {len(st.session_state.destination_route)}")

    def render_rescue_team_dashboard(self):
        """Render dashboard for rescue teams"""
        
        self.render_alert_box()

        if st.session_state.team_location is None:
            st.info("Set your team's start location using the 'Quick Action' button above to begin.")
            return

        if st.session_state.reporting_victim:
            victim = st.session_state.current_assignment
            st.subheader(f"Reporting SOS: {victim['name']}")
            st.info(f"You are at location **{victim['position']}**. Please report the victim's condition.")
            self.render_victim_report_form(victim)
        
        elif st.session_state.current_assignment is not None:
            victim = st.session_state.current_assignment
            st.subheader("Active Assignment")
            st.warning(f"Proceed to {victim['zone']} at **{victim['position']}** to investigate {victim['name']}.")
            
            if st.button("Arrived at Scene: Report Victim", use_container_width=True, type="primary"):
                st.session_state.reporting_victim = True
                st.rerun()
        
        else:
            st.subheader("Get Next Assignment")
            st.success(f"Your team is active and ready for dispatch from {st.session_state.team_location}.")
            if st.button("Get Next AI-Dispatched SOS Assignment", use_container_width=True, type="primary"):
                assignment = st.session_state.scenario.get_next_sos_assignment()
                if assignment:
                    st.session_state.current_assignment = assignment
                    st.rerun()
                else:
                    st.info("No active SOS alerts to assign at this time. Stand by.")

    def render_victim_report_form(self, victim_data: Dict[str, Any]):
        """Renders the form for a rescue team to report a victim's details."""
        
        with st.form(key=f"report_form_{victim_data['id']}"):
            name = st.text_input("Victim Name/ID", value=victim_data['name'])
            
            cond_list = ["critical", "serious", "stable", "unknown"]
            cond_index = cond_list.index(victim_data['condition']) if victim_data['condition'] in cond_list else 3
            condition = st.selectbox("Observed Condition", cond_list, index=cond_index)
            
            age_val = 30 
            if isinstance(victim_data['age'], int):
                age_val = victim_data['age']
            age = st.number_input("Victim Age (Approx.)", min_value=0, max_value=120, value=age_val)
            
            injured = st.checkbox("Visibly Injured?", value=True)
            
            hospital_options = list(st.session_state.scenario.hospitals.keys())
            hospital = st.selectbox("Assign to Hospital", hospital_options, index=0)
            
            notes = st.text_area("Additional Notes (e.g., 'head trauma', 'severe bleeding')")
            
            submitted = st.form_submit_button("Send Report to Hospital")
            
            if submitted:
                self.submit_victim_report(victim_data['id'], name, hospital, condition, age, injured, notes)

    def render_coordinator_dashboard(self):
        """Render dashboard for emergency coordinators"""
        st.subheader("Emergency Coordination Center")
        
        self.render_alert_box()
        
        st.info("AI Risk Assessment")
        rescue_success_prob = st.session_state.scenario.predict_rescue_success(
            st.session_state.weather_condition, st.session_state.collapse_risk
        )
        st.metric("Rescue Success Probability", f"{rescue_success_prob:.1%}")
        
        collapse_probs = st.session_state.scenario.predict_collapse_risk(st.session_state.weather_condition)
        resource_probs = st.session_state.scenario.predict_resource_status(
            st.session_state.weather_condition, st.session_state.collapse_risk
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Collapse Risk Probabilities:*")
            for risk, prob in collapse_probs.items():
                st.write(f"{risk}: {prob:.1%}")
        
        with col2:
            st.write("*Resource Status Probabilities:*")
            for status, prob in resource_probs.items():
                st.write(f"{status}: {prob:.1%}")
        
        st.info("Hospital Status")
        for hospital_name, hospital in st.session_state.scenario.hospitals.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{hospital_name}")
            with col2:
                st.write(f"Beds: {hospital['available_beds']}/{hospital['capacity']}")
            with col3:
                occupancy = (hospital['capacity'] - hospital['available_beds']) / hospital['capacity']
                st.progress(occupancy)
                
    def render_hospital_coordinator_dashboard(self):
        """Render dashboard for hospital coordinators"""
        self.render_alert_box()

        st.subheader("Incoming Patient Triage")
        
        if not st.session_state.hospital_messages:
            st.info("No incoming patients reported by rescue teams.")
        else:
            st.warning(f"**{len(st.session_state.hospital_messages)}** incoming patient reports:")
            
            priority_order = {'High (Red)': 0, 'Medium (Yellow)': 1, 'Low (Green)': 2}
            
            sorted_messages = sorted(
                st.session_state.hospital_messages, 
                key=lambda x: priority_order.get(x['priority'], 1) 
            )
            
            for report in sorted_messages:
                priority = report['priority']
                color = "blue"
                if priority == 'High (Red)': color = "red"
                if priority == 'Medium (Yellow)': color = "orange"
                if priority == 'Low (Green)': color = "green"

                with st.expander(f":{color}[AI Suggestion: {priority}] - Victim **{report['name']}** -> **{report['hospital']}**"):
                    st.write(f"**Condition:** {report['condition']} | **Age:** {report['age']} | **Injured:** {report['injured']}")
                    st.write(f"**Notes:** *{report['notes']}*")
                    
                    priority_list = ['High (Red)', 'Medium (Yellow)', 'Low (Green)']
                    current_index = priority_list.index(priority)
                    
                    st.selectbox(
                        "Confirm or Override Triage:",
                        priority_list,
                        index=current_index,
                        key=f"priority_{report['id']}",
                        on_change=self.update_patient_priority,
                        args=(report['id'],) 
                    )
        
        st.divider()
        
        st.subheader("Live Hospital Status")
        st.info("Monitor bed availability across the network.")
        
        for hospital_name, hospital in st.session_state.scenario.hospitals.items():
            st.write(f"**{hospital_name}**")
            col1, col2, col3 = st.columns(3)
            
            available = hospital['available_beds']
            capacity = hospital['capacity']
            occupied = capacity - available
            
            if capacity > 0:
                occupancy_pct = occupied / capacity
            else:
                occupancy_pct = 0
            
            col1.metric("Available Beds", f"{available} / {capacity}")
            col2.metric("Occupied Beds", f"{occupied}")
            col3.metric("Occupancy", f"{occupancy_pct:.0%}")
            
            if occupancy_pct > 0.8:
                st.error("Capacity Critical")
            elif occupancy_pct > 0.5:
                st.warning("High Occupancy")
            else:
                st.success("Nominal Occupancy")
            st.progress(occupancy_pct)

    def update_patient_priority(self, patient_id):
        """Callback to update a patient's priority in session state."""
        new_priority = st.session_state[f"priority_{patient_id}"]
        
        for patient in st.session_state.hospital_messages:
            if patient['id'] == patient_id:
                patient['priority'] = new_priority
                break

    def render_ai_coordination(self):
        """Render AI coordination panel for emergency coordinators"""
        if st.session_state.user_role not in ["emergency_coordinator"]:
            st.info("This section is only available for Emergency Coordinators")
            return
        
        st.subheader("AI-Powered Coordination Center")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Resource Management")
            
            st.write("*Manage Available Resources:*")
            
            st.metric("Medical Kits", st.session_state.resource_status['medical_kits'])
            st.metric("Transport Vehicles", st.session_state.resource_status['transport'])
            st.metric("Rescue Equipment", st.session_state.resource_status['equipment']) 
            
            st.subheader("Add Resources")
            add_kits = st.number_input("Add Medical Kits", 0, 100, 0)
            add_transport = st.number_input("Add Transport Vehicles", 0, 20, 0)
            add_equip = st.number_input("Add Rescue Equipment", 0, 50, 0) 
            
            if st.button("Add Resources"):
                st.session_state.resource_status['medical_kits'] += add_kits
                st.session_state.resource_status['transport'] += add_transport
                st.session_state.resource_status['equipment'] += add_equip 
                st.success("Resources added!")
                st.rerun()
            
            st.info("Volunteer Allocation")
            
            st.write("*Zone Demands:*")
            zone_demands = {
                'Northwest': {'urgency': 0.8, 'skills': {'medical': 3, 'rescue': 2, 'logistics': 1, 'communication': 1}},
                'Northeast': {'urgency': 0.6, 'skills': {'medical': 2, 'rescue': 1, 'logistics': 2, 'communication': 2}},
                'Southwest': {'urgency': 0.9, 'skills': {'medical': 4, 'rescue': 3, 'logistics': 1, 'communication': 1}},
                'Southeast': {'urgency': 0.7, 'skills': {'medical': 2, 'rescue': 2, 'logistics': 2, 'communication': 2}}
            }
            
            if st.button("Optimize Volunteer Allocation"):
                available_resources = {
                    'Northwest': st.session_state.resource_status,
                    'Northeast': st.session_state.resource_status,
                    'Southwest': st.session_state.resource_status,
                    'Southeast': st.session_state.resource_status
                }
                
                allocation = st.session_state.scenario.optimize_volunteer_allocation(
                    zone_demands, available_resources
                )
                st.session_state.volunteer_allocation = allocation
                st.success("Volunteer allocation optimized!")
                
                allocated_rescue = 0
                if allocation:
                    for zone, skills in allocation.items():
                        total_volunteers = sum(skills.values())
                        if total_volunteers > 0:
                            st.write(f"{zone} Zone:** {total_volunteers} volunteers")
                            for skill, count in skills.items():
                                if count > 0:
                                    st.write(f"  - {skill.title()}: {count}")
                                    if skill == 'rescue':
                                        allocated_rescue += count
                        else:
                            st.write(f"{zone} Zone:** No volunteers assigned")

                    st.session_state.resource_status['equipment'] -= allocated_rescue
                    st.warning(f"Depleted {allocated_rescue} rescue equipment units for this allocation.")
                    st.rerun()

        with col2:
            st.info("AI SOS Alert Triage")
            st.write("This panel tracks incoming SOS alerts and dispatches.")
            
            sos_alerts = [v for v in st.session_state.scenario.victims if "Citizen_SOS" in v['name'] and not v.get('rescued', False)]
            if sos_alerts:
                st.error(f"**{len(sos_alerts)} Active SOS Alerts**")
                for alert in sos_alerts[:5]:
                    status = "Pending"
                    if alert['assigned_team']:
                        status = "Assigned"
                    st.write(f"- SOS from {alert['zone']} at {alert['position']}. **Status: {status}**")
            else:
                st.success("No active SOS alerts.")

    def render_quick_status(self):
        """Render quick status panel"""
        st.subheader("Emergency Overview")
        
        scenario = st.session_state.scenario
        active_victims = len([v for v in scenario.victims if not v.get('rescued', False)])
        rescued_victims = len([v for v in scenario.victims if v.get('rescued', False)])
        
        st.write(f"*Disaster:* {scenario.scenario_type}")
        st.write(f"*Active Victims (Inital + SOS):* {active_victims}")
        st.write(f"*Rescued Victims (Reported):* {rescued_victims}")
        st.write(f"*Hospitals:* {len(scenario.hospitals)}")
        st.write(f"*Community Centers:* {len(scenario.community_centers)}")
        st.write(f"*Evacuation Shelters:* {len(scenario.evacuation_shelters)}")
        st.write(f"*Supply Depots:* {len(scenario.supply_depots)}")
        st.write(f"*Volunteers:* {len(scenario.volunteers)}")
        
        st.subheader("Conditions")
        st.write(f"*Weather:* {st.session_state.weather_condition}")
        st.write(f"*Risk Level:* {st.session_state.collapse_risk}")

    def render_emergency_map(self):
        """Render emergency map with hospitals and routes"""
        st.subheader("Live Emergency Map")
        
        if st.session_state.user_role == "citizen":
            st.info("Select a destination on the 'Dashboard' tab to see your route here.")
        else:
            st.info("This map shows the live status of all facilities, victims, and volunteers.")
        
        fig = self.create_emergency_map()
        st.plotly_chart(fig, use_container_width=True)

    def calculate_destination_route(self, destination_name, destination_info):
        """Calculate route to selected destination"""
        if not st.session_state.user_location:
            st.warning("Please set your location first")
            return
            
        route = st.session_state.scenario.find_evacuation_route(
            st.session_state.user_location, destination_info['position']
        )
        if route:
            st.session_state.selected_destination = destination_name
            st.session_state.destination_route = route
            st.success(f"Route found to {destination_name}: {len(route)} steps. View on 'Emergency Map' tab.")
        else:
            st.error(f"No safe route found to {destination_name}")

    def render_hospitals_zones(self):
        """Render hospitals and zones information"""
        st.subheader("Emergency Facilities & Resources")
        
        scenario = st.session_state.scenario
        
        st.info("Hospital Network")
        for hospital_name, hospital in scenario.hospitals.items():
            with st.expander(f"{hospital_name} - {hospital['available_beds']} beds available"):
                st.write(f"*Location:* Zone {scenario.get_zone_name(hospital['position'])}")
                st.write(f"*Coordinates:* {hospital['position']}")
                st.write(f"*Capacity:* {hospital['capacity']} beds")
                st.write(f"*Available:* {hospital['available_beds']} beds")
                
                if hospital['capacity'] > 0:
                    occupancy = ((hospital['capacity'] - hospital['available_beds']) / 
                                 hospital['capacity'] * 100)
                else:
                    occupancy = 0
                st.write(f"*Occupancy:* {occupancy:.1f}%")
        
        st.info("Community Centers")
        for center_name, center in scenario.community_centers.items():
            st.write(f"{center_name}:** {center['capacity']} people capacity")
            st.write(f"Location: {scenario.get_zone_name(center['position'])}")
        
        st.info("Evacuation Shelters")
        for shelter_name, shelter in scenario.evacuation_shelters.items():
            st.write(f"{shelter_name}:** {shelter['capacity']} people capacity")
            st.write(f"Location: {scenario.get_zone_name(shelter['position'])}")
        
        st.info("Supply Depots")
        for depot_name, depot in scenario.supply_depots.items():
            st.write(f"{depot_name}:** {depot['supplies']}")
            st.write(f"Location: {scenario.get_zone_name(depot['position'])}")

    def create_emergency_map(self):
        """Create emergency map with hospitals, victims, and routes"""
        scenario = st.session_state.scenario
        
        fig = go.Figure()
        
        if scenario.obstacles:
            obs_x, obs_y = zip(*scenario.obstacles)
            fig.add_trace(go.Scatter(
                x=obs_x, y=obs_y, mode='markers',
                marker=dict(color='gray', symbol='square', size=8, opacity=0.6),
                name='Obstacles',
                hoverinfo='skip'
            ))
        
        hospital_x, hospital_y, hospital_names = [], [], []
        for name, hospital in scenario.hospitals.items():
            hospital_x.append(hospital['position'][0])
            hospital_y.append(hospital['position'][1])
            hospital_names.append(name)
        
        fig.add_trace(go.Scatter(
            x=hospital_x, y=hospital_y, mode='markers+text',
            marker=dict(color='red', symbol='star', size=20),
            text=hospital_names,
            textposition="top center",
            name='Hospitals',
            hovertemplate='<b>%{text}</b><br>Available Beds: %{customdata}',
            customdata=[h['available_beds'] for h in scenario.hospitals.values()]
        ))
        
        center_x, center_y, center_names = [], [], []
        for name, center in scenario.community_centers.items():
            center_x.append(center['position'][0])
            center_y.append(center['position'][1])
            center_names.append(name)
        
        fig.add_trace(go.Scatter(
            x=center_x, y=center_y, mode='markers+text',
            marker=dict(color='green', symbol='circle', size=15),
            text=center_names,
            textposition="top center",
            name='Community Centers',
            hovertemplate='<b>%{text}</b>'
        ))
        
        shelter_x, shelter_y, shelter_names = [], [], []
        for name, shelter in scenario.evacuation_shelters.items():
            shelter_x.append(shelter['position'][0])
            shelter_y.append(shelter['position'][1])
            shelter_names.append(name)
        
        fig.add_trace(go.Scatter(
            x=shelter_x, y=shelter_y, mode='markers+text',
            marker=dict(color='orange', symbol='square', size=15),
            text=shelter_names,
            textposition="top center",
            name='Evacuation Shelters',
            hovertemplate='<b>%{text}</b>'
        ))
        
        depot_x, depot_y, depot_names = [], [], []
        for name, depot in scenario.supply_depots.items():
            depot_x.append(depot['position'][0])
            depot_y.append(depot['position'][1])
            depot_names.append(name)
        
        fig.add_trace(go.Scatter(
            x=depot_x, y=depot_y, mode='markers+text',
            marker=dict(color='purple', symbol='diamond', size=15),
            text=depot_names,
            textposition="top center",
            name='Supply Depots',
            hovertemplate='<b>%{text}</b>'
        ))
        
        active_victims = [v for v in scenario.victims if not v.get('rescued', False)]
        if active_victims:
            vic_x, vic_y, vic_colors, vic_conditions = [], [], [], []
            for victim in active_victims:
                vic_x.append(victim['position'][0])
                vic_y.append(victim['position'][1])
                
                if victim['condition'] == 'critical':
                    vic_colors.append('red')
                elif victim['condition'] == 'serious':
                    vic_colors.append('orange')
                elif victim['condition'] == 'unknown':
                    vic_colors.append('magenta') 
                else:
                    vic_colors.append('yellow')
                
                vic_conditions.append(victim['condition'])
            
            fig.add_trace(go.Scatter(
                x=vic_x, y=vic_y, mode='markers',
                marker=dict(color=vic_colors, symbol='circle', size=10),
                name='Active Victims',
                hovertemplate='<b>Victim</b><br>Condition: %{customdata}',
                customdata=vic_conditions
            ))
        
        if scenario.volunteers:
            vol_x, vol_y, vol_skills = [], [], []
            for volunteer in scenario.volunteers:
                vol_x.append(volunteer['position'][0])
                vol_y.append(volunteer['position'][1])
                vol_skills.append(volunteer['skill'])
            
            fig.add_trace(go.Scatter(
                x=vol_x, y=vol_y, mode='markers',
                marker=dict(color='blue', symbol='triangle-up', size=12),
                name='Volunteers',
                hovertemplate='<b>Volunteer</b><br>Skill: %{customdata}',
                customdata=vol_skills
            ))
        
        if st.session_state.user_location:
            fig.add_trace(go.Scatter(
                x=[st.session_state.user_location[0]], 
                y=[st.session_state.user_location[1]], 
                mode='markers+text',
                marker=dict(color='darkblue', symbol='triangle-up', size=25),
                text=["Your Location"],
                textposition="top center",
                name='Your Location'
            ))
            
            if st.session_state.destination_route and st.session_state.selected_destination:
                route_x, route_y = zip(*st.session_state.destination_route)
                fig.add_trace(go.Scatter(
                    x=route_x, y=route_y, mode='lines+markers',
                    line=dict(color='green', width=3, dash='solid'),
                    marker=dict(size=6),
                    name=f'Route to {st.session_state.selected_destination}'
                ))
        
        zone_boundaries = [
            [(0, 0), (9, 0), (9, 9), (0, 9)],  # NW
            [(10, 0), (19, 0), (19, 9), (10, 9)],  # NE
            [(0, 10), (9, 10), (9, 19), (0, 19)],  # SW
            [(10, 10), (19, 10), (19, 19), (10, 19)]  # SE
        ]
        
        zone_names = ["Northwest", "Northeast", "Southwest", "Southeast"]
        zone_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        
        for i, boundary in enumerate(zone_boundaries):
            x, y = zip(*boundary)
            x = list(x) + [x[0]] 
            y = list(y) + [y[0]]
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill="toself",
                fillcolor=zone_colors[i],
                line=dict(color='black', width=1),
                opacity=0.2,
                name=f'{zone_names[i]} Zone',
                showlegend=False
            ))
            
            fig.add_annotation(
                x=sum([p[0] for p in boundary])/4,
                y=sum([p[1] for p in boundary])/4,
                text=zone_names[i],
                showarrow=False,
                font=dict(size=12, color="black")
            )
        
        fig.update_layout(
            title="Emergency Response Map - All Facilities & Evacuation Routes",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    # --- Action implementations ---
    def set_user_location(self):
        """Set user location and assign zone"""
        scenario = st.session_state.scenario
        st.session_state.user_location = scenario.get_random_free_position()
        st.session_state.user_zone = scenario.get_zone_name(st.session_state.user_location)
        st.session_state.selected_destination = None
        st.session_state.destination_route = None
        st.session_state.sos_sent_message = None 
        
        st.success(f"Location set to: {st.session_state.user_zone}")
        st.info(f"Coordinates: {st.session_state.user_location}")

    def calculate_safety_status(self):
        """Calculate safety status based on proximity to obstacles"""
        if not st.session_state.user_location:
            return "unknown"
        
        user_pos = st.session_state.user_location
        scenario = st.session_state.scenario
        
        if scenario.obstacles:
            min_obstacle_dist = min(
                abs(user_pos[0] - obs[0]) + abs(user_pos[1] - obs[1])
                for obs in scenario.obstacles
            )
        else:
            min_obstacle_dist = float('inf')
        
        if min_obstacle_dist <= 2:
            return "danger"
        elif min_obstacle_dist <= 5:
            return "warning"
        else:
            return "safe"

    def send_sos_alert(self):
        """Handle SOS alert from a citizen"""
        if not st.session_state.user_location:
            st.warning("Please set your location first before sending an SOS")
            return
        
        for victim in st.session_state.scenario.victims:
            if victim['position'] == st.session_state.user_location and "Citizen_SOS" in victim['name']:
                st.warning("An SOS has already been sent from your location. A rescue team is being prioritized.")
                st.session_state.sos_sent_message = "An SOS has already been sent from your location. A rescue team is being prioritized."
                return

        safety_status = self.calculate_safety_status()
        in_danger = True if safety_status in ['danger', 'warning'] else False
        victim_id = len(st.session_state.scenario.victims) + 1
        
        new_victim = {
            'id': victim_id,
            'name': f'Citizen_SOS_{victim_id}',
            'position': st.session_state.user_location,
            'zone': st.session_state.user_zone,
            'injured': 'unknown',
            'age': 'unknown',
            'condition': 'unknown',
            'in_danger_zone': in_danger,
            'special_needs': False,
            'rescued': False,
            'assigned_hospital': None,
            'priority_score': 0,
            'assigned_team': None, 
        }
        
        st.session_state.scenario.victims.append(new_victim)
        
        st.session_state.alerts.append(
            f"New SOS Alert from {st.session_state.user_zone} at {st.session_state.user_location}"
        )
        
        st.session_state.sos_sent_message = "SOS Alert sent! An alert has been sent to the emergency coordinator."
        st.rerun()

    def submit_victim_report(self, victim_id, name, hospital, condition, age, injured, notes):
        """Submit an SOS victim report from the field"""
        
        if st.session_state.scenario.hospitals[hospital]['available_beds'] <= 0:
            st.error(f"{hospital} has no available beds! Please assign to another hospital.")
            return

        st.session_state.scenario.hospitals[hospital]['available_beds'] -= 1
        
        if injured:
            if st.session_state.resource_status['medical_kits'] > 0:
                st.session_state.resource_status['medical_kits'] -= 1
            else:
                st.warning("Report sent, but WARNING: No medical kits were available!")
        
        if st.session_state.resource_status['transport'] > 0:
            st.session_state.resource_status['transport'] -= 1
        else:
            st.warning("Report sent, but WARNING: No transport vehicles were available!")

        raw_report = {'condition': condition, 'age': age, 'notes': notes}
        ai_priority = st.session_state.scenario.triage_ai.suggest_priority(raw_report)
        
        report_data = {
            'id': f"patient_{victim_id}",
            'name': name,
            'hospital': hospital,
            'condition': condition,
            'age': age,
            'injured': injured,
            'notes': notes,
            'priority': ai_priority 
        }
        st.session_state.hospital_messages.append(report_data)
        
        for victim in st.session_state.scenario.victims:
            if victim['id'] == victim_id:
                victim['name'] = name
                victim['condition'] = condition
                victim['age'] = age
                victim['injured'] = injured
                victim['rescued'] = True
                victim['assigned_hospital'] = hospital
                break
        
        st.session_state.current_assignment = None
        st.session_state.reporting_victim = False
        
        st.success("Report sent to hospital coordinator and bed reserved!")
        st.rerun()

    def allocate_volunteers(self):
        """Allocate volunteers to zones"""
        zone_demands = {
            'Northwest': {'urgency': 0.8, 'skills': {'medical': 3, 'rescue': 2, 'logistics': 1, 'communication': 1}},
            'Northeast': {'urgency': 0.6, 'skills': {'medical': 2, 'rescue': 1, 'logistics': 2, 'communication': 2}},
            'Southwest': {'urgency': 0.9, 'skills': {'medical': 4, 'rescue': 3, 'logistics': 1, 'communication': 1}},
            'Southeast': {'urgency': 0.7, 'skills': {'medical': 2, 'rescue': 2, 'logistics': 2, 'communication': 2}}
        }
        
        available_resources = {
            'Northwest': st.session_state.resource_status,
            'Northeast': st.session_state.resource_status,
            'Southwest': st.session_state.resource_status,
            'Southeast': st.session_state.resource_status
        }
        
        allocation = st.session_state.scenario.optimize_volunteer_allocation(
            zone_demands, available_resources
        )
        st.session_state.volunteer_allocation = allocation
        st.success("👥 Volunteers allocated optimally!")
        
        allocated_rescue = 0
        if allocation:
            for zone, skills in allocation.items():
                total_volunteers = sum(skills.values())
                if total_volunteers > 0:
                    st.write(f"{zone} Zone:** {total_volunteers} volunteers")
                    for skill, count in skills.items():
                        if count > 0:
                            st.write(f"  - {skill.title()}: {count}")
                            if skill == 'rescue':
                                allocated_rescue += count
                else:
                    st.write(f"{zone} Zone:** No volunteers assigned")

            st.session_state.resource_status['equipment'] -= allocated_rescue
            st.warning(f"Depleted {allocated_rescue} rescue equipment units for this allocation.")
            st.rerun()

    def show_system_status(self):
        """Show system status"""
        st.info("Emergency Response System Status")
        
        scenario = st.session_state.scenario
        active_victims = len([v for v in scenario.victims if not v.get('rescued', False)])
        rescued_victims = len([v for v in scenario.victims if v.get('rescued', False)])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Victims (Inital + SOS)", active_victims)
            st.metric("Rescued Victims (Reported)", rescued_victims)
        
        with col2:
            st.metric("Hospital Capacity", 
                      sum(h['capacity'] for h in scenario.hospitals.values()))
            st.metric("Available Beds",
                      sum(h['available_beds'] for h in scenario.hospitals.values()))
        
        with col3:
            rescue_prob = scenario.predict_rescue_success(
                st.session_state.weather_condition, st.session_state.collapse_risk
            )
            st.metric("Rescue Success Probability", f"{rescue_prob:.1%}")

    def send_emergency_alert(self):
        """Send emergency alert"""
        alert_message = st.text_area("Alert Message", 
                                     "Emergency situation - proceed to nearest safe zone or hospital. Follow official instructions.")
        
        if st.button("Broadcast Emergency Alert"):
            st.session_state.alerts.append(f"OFFICIAL ALERT: {alert_message}")
            st.success("Emergency alert broadcasted to all zones!")

# Run the application
if __name__ == "__main__":
    ai_agent = DisasterResponseAI()
    ai_agent.run()