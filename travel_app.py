import streamlit as st
import requests
from datetime import datetime, timedelta

# API URLs
API_BASE_URL = "http://localhost:8000"
API_URL_FLIGHTS = f"{API_BASE_URL}/search_flights/"
API_URL_HOTELS = f"{API_BASE_URL}/search_hotels/"
API_URL_ITINERARY = f"{API_BASE_URL}/generate_itinerary/"

# App title
st.title("AI Travel Planner")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Flights", "Hotels", "Itinerary"])

with tab1:
    st.header("Find Flights")
    origin = st.text_input("Origin (Airport Code)", "JFK")
    destination = st.text_input("Destination (Airport Code)", "LAX")
    
    col1, col2 = st.columns(2)
    with col1:
        outbound_date = st.date_input("Departure Date", datetime.now() + timedelta(days=30))
    with col2:
        return_date = st.date_input("Return Date", datetime.now() + timedelta(days=37))
    
    if st.button("Search Flights"):
        # Validate dates
        if outbound_date > return_date:
            st.error("⚠️ Departure date must be before return date. Please adjust your dates.")
        else:
            with st.spinner("Searching for flights..."):
                try:
                    response = requests.post(
                        API_URL_FLIGHTS,
                        json={
                            "origin": origin,
                            "destination": destination,
                            "outbound_date": outbound_date.strftime("%Y-%m-%d"),
                            "return_date": return_date.strftime("%Y-%m-%d")
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Display flights
                        st.subheader("Available Flights")
                        if data.get("flights"):
                            for i, flight in enumerate(data["flights"], 1):
                                st.write(f"**Flight {i}**")
                                st.write(f"- Airline: {flight.get('airline', 'Unknown')}")
                                st.write(f"- Price: {flight.get('price', 'Unknown')}")
                                st.write(f"- Duration: {flight.get('duration', 'Unknown')}")
                                st.write(f"- Stops: {flight.get('stops', 'Unknown')}")
                                st.write(f"- Departure: {flight.get('departure', 'Unknown')}")
                                st.write(f"- Arrival: {flight.get('arrival', 'Unknown')}")
                                st.write(f"- Class: {flight.get('travel_class', 'Unknown')}")
                                st.write("---")
                        else:
                            st.write("No flights found for this route and dates.")
                        
                        # Display AI recommendation
                        st.subheader("AI Recommendation")
                        st.markdown(data.get("ai_flight_recommendation", "No recommendation available"))
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
                    st.info("This could be due to an issue with the SerpAPI service or your API key. Try again later or check your backend logs for more details.")

with tab2:
    st.header("Find Hotels")
    location = st.text_input("Destination", "New York")
    
    col1, col2 = st.columns(2)
    with col1:
        check_in_date = st.date_input("Check-in Date", datetime.now() + timedelta(days=30))
    with col2:
        check_out_date = st.date_input("Check-out Date", datetime.now() + timedelta(days=37))
    
    if st.button("Search Hotels"):
        with st.spinner("Searching for hotels..."):
            try:
                response = requests.post(
                    API_URL_HOTELS,
                    json={
                        "location": location,
                        "check_in_date": check_in_date.strftime("%Y-%m-%d"),
                        "check_out_date": check_out_date.strftime("%Y-%m-%d")
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    # Display hotels
                    st.subheader("Available Hotels")
                    if data.get("hotels"):
                        for i, hotel in enumerate(data["hotels"], 1):
                            st.write(f"**Hotel {i}**")
                            st.write(f"- Name: {hotel.get('name', 'Unknown')}")
                            st.write(f"- Price: {hotel.get('price', 'Unknown')}")
                            st.write(f"- Rating: {hotel.get('rating', 'Unknown')}/10")
                            st.write(f"- Location: {hotel.get('location', 'Unknown')}")
                            st.write("---")
                    else:
                        st.write("No hotels found for this location and dates.")
                    
                    # Display AI recommendation
                    st.subheader("AI Recommendation")
                    st.markdown(data.get("ai_hotel_recommendation", "No recommendation available"))
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("This could be due to an issue with the SerpAPI service or your API key. Try again later or check your backend logs for more details.")

with tab3:
    st.header("Generate Itinerary")
    destination = st.text_input("Destination for Itinerary", "Paris")
    
    col1, col2 = st.columns(2)
    with col1:
        check_in_date = st.date_input("Arrival Date", datetime.now() + timedelta(days=30), key="itin_check_in")
    with col2:
        check_out_date = st.date_input("Departure Date", datetime.now() + timedelta(days=37), key="itin_check_out")
    
    flights_info = st.text_area("Flight Details (Optional)", "Flight XYZ123, Departing JFK at 18:00, Arriving CDG at 07:30")
    hotels_info = st.text_area("Hotel Details (Optional)", "Grand Hotel Paris, Located in the 8th arrondissement")
    
    if st.button("Generate Itinerary"):
        with st.spinner("Generating your personalized itinerary..."):
            try:
                response = requests.post(
                    API_URL_ITINERARY,
                    json={
                        "destination": destination,
                        "check_in_date": check_in_date.strftime("%Y-%m-%d"),
                        "check_out_date": check_out_date.strftime("%Y-%m-%d"),
                        "flights": flights_info,
                        "hotels": hotels_info
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    # Display itinerary
                    st.subheader("Your Personalized Itinerary")
                    st.markdown(data.get("itinerary", "No itinerary available"))
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("This could be due to an issue with the SerpAPI service or your API key. Try again later or check your backend logs for more details.")