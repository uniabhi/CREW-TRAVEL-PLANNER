import os
import uvicorn
import asyncio
import logging
from serpapi.google_search import GoogleSearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from crewai import Agent, Task, Crew, Process, LLM
from datetime import datetime
from functools import lru_cache

# Load API Keys
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY")
# Check if API keys are set

# Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def initialize_llm():
    """Initialize and cache the LLM instance to avoid repeated initializations."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=GEMINI_API_KEY
    )

class FlightRequest(BaseModel):
    origin: str
    destination: str
    outbound_date: str
    return_date: str

class HotelRequest(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str

class ItineraryRequest(BaseModel):
    destination: str
    check_in_date: str
    check_out_date: str
    flights: str
    hotels: str

class FlightInfo(BaseModel):
    airline: str = "Unknown Airline"
    price: str = "Unknown"
    duration: str = "Unknown"
    stops: str = "Unknown"
    departure: str = "Unknown"
    arrival: str = "Unknown"
    travel_class: str = "Economy"
    return_date: str = "Unknown"
    airline_logo: str = ""

class HotelInfo(BaseModel):
    name: str
    price: str = "Not available"
    rating: float = 0.0
    location: str = "Unknown Location"
    link: str = ""

class AIResponse(BaseModel):
    flights: List[FlightInfo] = []
    hotels: List[HotelInfo] = []
    ai_flight_recommendation: str = ""
    ai_hotel_recommendation: str = ""
    itinerary: str = ""

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Travel Planning API", version="1.0.1")

async def run_search(params):
    """Generic function to run SerpAPI searches asynchronously."""
    try:
        logger.info(f"Sending search request with parameters: {params}")
        result = await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
        logger.info(f"Search result keys: {result.keys()}")
        return result
    except Exception as e:
        logger.exception(f"SerpAPI search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")


async def search_flights(flight_request: FlightRequest):
    """Fetch real-time flight details from Google Flights using SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} to {flight_request.destination}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": flight_request.origin.strip().upper(),
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "USD"
    }

    try:
        search_results = await run_search(params)
        
        # Extract and combine flights from best_flights and other_flights
        all_flights = []
        
        # Add best flights if available
        if "best_flights" in search_results and search_results["best_flights"]:
            best_flights = search_results.get("best_flights", [])
            if isinstance(best_flights, list):
                all_flights.extend(best_flights)
        
        # Add other flights if available
        if "other_flights" in search_results and search_results["other_flights"]:
            other_flights = search_results.get("other_flights", [])
            if isinstance(other_flights, list):
                all_flights.extend(other_flights)
        
        # Log a sample flight for debugging
        if all_flights and len(all_flights) > 0:
            logger.info(f"Sample flight data: {all_flights[0]}")
        
        # Transform the flight data to match your model
        transformed_flights = transform_flight_data(all_flights)
        
        # If no flights were found, return an empty list
        if not transformed_flights:
            logger.warning("No flights found in the search results")
            return []
            
        return transformed_flights
    except Exception as e:
        logger.exception(f"Error searching for flights: {str(e)}")
        # Return an empty list instead of None
        return []

def get_mock_hotel_data(location):
    """Generate mock hotel data when API doesn't provide enough information."""
    return [
        HotelInfo(
            name=f"Luxury Hotel {location}",
            price="$199 per night",
            rating=9.2,
            location=f"Downtown {location}",
            link=""
        ),
        HotelInfo(
            name=f"Plaza Hotel {location}",
            price="$249 per night",
            rating=8.8,
            location=f"City Center, {location}",
            link=""
        ),
        HotelInfo(
            name=f"Budget Inn {location}",
            price="$129 per night",
            rating=7.6,
            location=f"Airport Area, {location}",
            link=""
        ),
        HotelInfo(
            name=f"Boutique Hotel {location}",
            price="$179 per night",
            rating=8.9,
            location=f"Historic District, {location}",
            link=""
        ),
        HotelInfo(
            name=f"Grand Suites {location}",
            price="$289 per night",
            rating=9.5,
            location=f"Riverfront, {location}",
            link=""
        )
    ]

async def search_hotels(hotel_request: HotelRequest):
    """Fetch hotel information from SerpAPI."""
    logger.info(f"Searching hotels for: {hotel_request.location}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "en",
        "gl": "us",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "USD",
        "sort_by": 3,
        "rating": 8
    }

    try:
        search_results = await run_search(params)
        
        # Log the complete response structure
        logger.info(f"Search result keys: {list(search_results.keys())}")
        
        # Check if properties exist and log a sample
        hotels = search_results.get("properties", [])
        if hotels and len(hotels) > 0:
            logger.info(f"First hotel keys: {list(hotels[0].keys())}")
            
            # Optional: log specific fields we're interested in
            if "price" in hotels[0]:
                logger.info(f"Price example: {hotels[0]['price']}")
            if "rating" in hotels[0]:
                logger.info(f"Rating example: {hotels[0]['rating']}")
            if "address" in hotels[0]:
                logger.info(f"Address example: {hotels[0]['address']}")
        
        # Transform hotels to match our model
        transformed_hotels = transform_hotel_data(hotels)
        
        # If all hotels have default values, use mock data instead
        has_valid_data = any(
            hotel.price != "Not available" or 
            hotel.rating > 0.0 or 
            hotel.location != "Unknown Location" 
            for hotel in transformed_hotels
        )
        
        if not transformed_hotels or not has_valid_data:
            logger.info("Using mock hotel data due to insufficient API data")
            return get_mock_hotel_data(hotel_request.location)
            
        return transformed_hotels
    except Exception as e:
        logger.exception(f"Error searching for hotels: {str(e)}")
        # Return mock data on error
        logger.info("Using mock hotel data due to API error")
        return get_mock_hotel_data(hotel_request.location)

async def get_ai_recommendation(data_type, formatted_data):
    logger.info(f"Getting {data_type} analysis from AI")
    llm_model = initialize_llm()

    # Configure agent based on data type
    if data_type == "flights":
        role = "AI Flight Analyst"
        goal = "Analyze flight options and recommend the best one considering price, duration, stops, and overall convenience."
        backstory = f"AI expert that provides in-depth analysis comparing flight options based on multiple factors."
        description = """
        Recommend the best flight from the available options, based on the details provided below:

        **Reasoning for Recommendation:**
        - **Price:** Provide a detailed explanation about why this flight offers the best value compared to others.
        - **Duration:** Explain why this flight has the best duration in comparison to others.
        - **Stops:** Discuss why this flight has minimal or optimal stops.
        - **Travel Class:** Describe why this flight provides the best comfort and amenities.

        Use the provided flight data as the basis for your recommendation. Be sure to justify your choice using clear reasoning for each attribute. Do not repeat the flight details in your response.
        """
    elif data_type == "hotels":
        role = "AI Hotel Analyst"
        goal = "Analyze hotel options and recommend the best one considering price, rating, location, and amenities."
        backstory = f"AI expert that provides in-depth analysis comparing hotel options based on multiple factors."
        description = """
        Based on the following analysis, generate a detailed recommendation for the best hotel. Your response should include clear reasoning based on price, rating, location, and amenities.

        **AI Hotel Recommendation**
        We recommend the best hotel based on the following analysis:

        **Reasoning for Recommendation**:
        - **Price:** The recommended hotel is the best option for the price compared to others, offering the best value for the amenities and services provided.
        - **Rating:** With a higher rating compared to the alternatives, it ensures a better overall guest experience. Explain why this makes it the best choice.
        - **Location:** The hotel is in a prime location, close to important attractions, making it convenient for travelers.
        - **Amenities:** The hotel offers amenities like Wi-Fi, pool, fitness center, free breakfast, etc. Discuss how these amenities enhance the experience, making it suitable for different types of travelers.

        **Reasoning Requirements**:
        - Ensure that each section clearly explains why this hotel is the best option based on the factors of price, rating, location, and amenities.
        - Compare it against the other options and explain why this one stands out.
        - Provide concise, well-structured reasoning to make the recommendation clear to the traveler.
        - Your recommendation should help a traveler make an informed decision based on multiple factors, not just one.
        """
    else:
        raise ValueError("Invalid data type for AI recommendation")

    # Create the agent and task
    analyze_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_model,
        verbose=False
    )

    analyze_task = Task(
        description=f"{description}\n\nData to analyze:\n{formatted_data}",
        agent=analyze_agent,
        expected_output=f"A structured recommendation explaining the best {data_type} choice based on the analysis of provided details."
    )

    # Define CrewAI Workflow for the agent
    analyst_crew = Crew(
        agents=[analyze_agent],
        tasks=[analyze_task],
        process=Process.sequential,
        verbose=False
    )

    # Execute CrewAI Process
    crew_results = await asyncio.to_thread(analyst_crew.kickoff)
    return str(crew_results)

async def generate_itinerary(destination, flights_text, hotels_text, check_in_date, check_out_date):
    """Generate a detailed travel itinerary based on flight and hotel information."""
    try:
        # Convert the string dates to datetime objects
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")

        # Calculate the difference in days
        days = (check_out - check_in).days

        llm_model = initialize_llm()
        analyze_agent = Agent(
            role="AI Travel Planner",
            goal="Create a detailed itinerary for the user based on flight and hotel information",
            backstory="AI travel expert generating a day-by-day itinerary including flight details, hotel stays, and must-visit locations in the destination.",
            llm=llm_model,
            verbose=False
        )

        analyze_task = Task(
            description=f"""
                Based on the following details, create a {days}-day itinerary for the user:
                **Flight Details**: {flights_text}
                **Hotel Details**: {hotels_text}
                **Destination**: {destination}
                **Travel Dates**: {check_in_date} to {check_out_date} ({days} days)

                The itinerary should include:
                - Flight arrival and departure information
                - Hotel check-in and check-out details
                - Day-by-day breakdown of activities
                - Must-visit attractions and estimated visit times
                - Restaurant recommendations for meals
                - Tips for local transportation

                **Format Requirements**:
                - Use markdown formatting with clear headings (# for main headings, ## for days, ### for sections)
                - Include emojis for different types of activities (🏛️ for landmarks, 🍽️ for restaurants, etc.)
                - Use bullet points for listing activities
                - Include estimated timings for each activity
                - Format the itinerary to be visually appealing and easy to read
            """,
            agent=analyze_agent,
            expected_output="A well-structured, visually appealing itinerary in markdown format, including flight, hotel, and day-wise breakdown with emojis, headers, and bullet points."
        )

        itinerary_planner_crew = Crew(
            agents=[analyze_agent],
            tasks=[analyze_task],
            process=Process.sequential,
            verbose=False
        )

        crew_results = await asyncio.to_thread(itinerary_planner_crew.kickoff)
        return str(crew_results)
    except Exception as e:
        # Handle any exceptions
        return f"Error generating itinerary: {str(e)}"
    
def format_travel_data(data_type, data):
    """Format travel data (flights or hotels) for AI analysis."""
    formatted_data = ""
    
    if data is None:
        return f"No {data_type} data available. Please check your search parameters or try again later."
    
    if data_type == "flights":
        formatted_data = "**Available Flights:**\n\n"
        for i, flight in enumerate(data, 1):
            formatted_data += f"**Flight {i}:**\n"
            # Check if we're dealing with a FlightInfo object or a dictionary
            if hasattr(flight, 'airline'):
                # It's a FlightInfo object
                formatted_data += f"- Airline: {flight.airline}\n"
                formatted_data += f"- Price: {flight.price}\n"
                formatted_data += f"- Duration: {flight.duration}\n"
                formatted_data += f"- Stops: {flight.stops}\n"
                formatted_data += f"- Departure: {flight.departure}\n"
                formatted_data += f"- Arrival: {flight.arrival}\n"
                formatted_data += f"- Class: {flight.travel_class}\n\n"
            else:
                # It's a dictionary
                formatted_data += f"- Airline: {flight.get('airline', 'Unknown')}\n"
                formatted_data += f"- Price: {flight.get('price', 'Unknown')}\n"
                formatted_data += f"- Duration: {flight.get('duration', 'Unknown')}\n"
                formatted_data += f"- Stops: {flight.get('stops', 'Unknown')}\n"
                formatted_data += f"- Departure: {flight.get('departure', 'Unknown')}\n"
                formatted_data += f"- Arrival: {flight.get('arrival', 'Unknown')}\n"
                formatted_data += f"- Class: {flight.get('travel_class', 'Unknown')}\n\n"
    
    elif data_type == "hotels":
        formatted_data = "**Available Hotels:**\n\n"
        for i, hotel in enumerate(data, 1):
            formatted_data += f"**Hotel {i}:**\n"
            # Check if we're dealing with a HotelInfo object or a dictionary
            if hasattr(hotel, 'name'):
                # It's a HotelInfo object
                formatted_data += f"- Name: {hotel.name}\n"
                formatted_data += f"- Price: {hotel.price}\n"
                formatted_data += f"- Rating: {hotel.rating}\n"
                formatted_data += f"- Location: {hotel.location}\n\n"
            else:
                # It's a dictionary
                formatted_data += f"- Name: {hotel.get('name', 'Unknown')}\n"
                formatted_data += f"- Price: {hotel.get('price', 'Unknown')}\n"
                formatted_data += f"- Rating: {hotel.get('rating', 'Unknown')}\n"
                formatted_data += f"- Location: {hotel.get('location', 'Unknown')}\n\n"
    
    return formatted_data

def transform_flight_data(flights_data):
    """Transform SerpAPI flight data to match our FlightInfo model."""
    if not flights_data:
        return []
        
    transformed_flights = []
    for flight_data in flights_data:
        try:
            # Extract nested flight information
            flights = flight_data.get('flights', [])
            if not flights and isinstance(flight_data, dict):
                # If there's no 'flights' key, treat the entire object as a single flight
                flights = [flight_data]
            
            for flight in flights:
                # Get departure and arrival info
                departure_info = flight.get('departure_airport', {})
                arrival_info = flight.get('arrival_airport', {})
                
                # Convert price to string if it's an integer
                price = flight_data.get('price', 'Unknown')
                if isinstance(price, int):
                    price = f"${price}"
                
                # Create flight info object with default values for missing fields
                flight_info = FlightInfo(
                    airline=flight.get('airline', flight_data.get('airline', 'Unknown Airline')),
                    price=price,
                    duration=str(flight.get('duration', flight_data.get('duration', 'Unknown'))),
                    stops=str(flight_data.get('stops', 'Nonstop')),
                    departure=departure_info.get('time', 'Unknown'),
                    arrival=arrival_info.get('time', 'Unknown'),
                    travel_class=flight.get('travel_class', flight_data.get('travel_class', 'Economy')),
                    return_date=flight_data.get('return_date', 'Unknown'),
                    airline_logo=flight.get('airline_logo', flight_data.get('airline_logo', ''))
                )
                transformed_flights.append(flight_info)
        except Exception as e:
            logger.error(f"Error transforming flight data: {str(e)}")
            # Continue with next flight if one fails
            continue
    
    return transformed_flights

def transform_hotel_data(hotels_data):
    """Transform SerpAPI hotel data to match our HotelInfo model."""
    if not hotels_data:
        return []
        
    transformed_hotels = []
    for hotel in hotels_data:
        try:
            # Log the structure of a sample hotel
            if len(transformed_hotels) == 0:
                logger.info(f"Sample hotel data structure: {hotel.keys()}")
                logger.info(f"Sample hotel data: {hotel}")
            
            # Try to extract price from different possible locations
            price = "Not available"
            if "price" in hotel:
                price = hotel["price"]
            elif "price_overview" in hotel:
                price = hotel["price_overview"]
            elif "pricing" in hotel and hotel["pricing"] is not None:
                price = hotel["pricing"]
            
            # Try to extract rating from different possible locations
            rating = 0.0
            if "rating" in hotel and hotel["rating"] is not None:
                try:
                    rating = float(hotel["rating"])
                except (ValueError, TypeError):
                    pass
            elif "stars" in hotel and hotel["stars"] is not None:
                try:
                    rating = float(hotel["stars"])
                except (ValueError, TypeError):
                    pass
            
            # Try to extract location information
            location = "Unknown Location"
            if "address" in hotel and hotel["address"] is not None:
                location = hotel["address"]
            elif "address_info" in hotel and hotel["address_info"] is not None:
                location = hotel["address_info"].get("street", "Unknown Street")
                city = hotel["address_info"].get("city", "")
                if city:
                    location += f", {city}"
            elif "location" in hotel and hotel["location"] is not None:
                location = hotel["location"]
            elif "neighborhood" in hotel and hotel["neighborhood"] is not None:
                location = hotel["neighborhood"]
            
            # Create hotel info object
            hotel_info = HotelInfo(
                name=hotel.get("name", "Unknown Hotel"),
                price=price,
                rating=rating,
                location=location,
                link=hotel.get("link", "")
            )
            transformed_hotels.append(hotel_info)
        except Exception as e:
            logger.error(f"Error transforming hotel data: {str(e)}")
            continue
    
    return transformed_hotels

@app.get("/")
async def root():
    return {"message": "Welcome to the Travel Planning API. Use the endpoints to search for flights, hotels, and generate itineraries."}

@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API is running smoothly."}  

@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(flight_request: FlightRequest):
    flights = await search_flights(flight_request)
    flights_text = format_travel_data("flights", flights)
    
    # If no flights were found, return a message instead of trying to get a recommendation
    if not flights:
        return AIResponse(
            flights=[], 
            ai_flight_recommendation="No flights found for the specified route and dates. Please try different search parameters."
        )
    
    ai_recommendation = await get_ai_recommendation("flights", flights_text)
    return AIResponse(flights=flights, ai_flight_recommendation=ai_recommendation)

@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(hotel_request: HotelRequest):
    try:
        hotels = await search_hotels(hotel_request)
        
        # Format hotel data for AI analysis
        hotel_text_data = "\n\n".join([
            f"**Hotel {i+1}:** {hotel.name}\n- Price: {hotel.price}\n- Rating: {hotel.rating}\n- Location: {hotel.location}"
            for i, hotel in enumerate(hotels)
        ])
        
        ai_recommendation = await get_ai_recommendation("hotels", hotel_text_data)
        return AIResponse(hotels=hotels, ai_hotel_recommendation=ai_recommendation)
    except Exception as e:
        logger.exception(f"Error in hotel recommendations: {str(e)}")
        return AIResponse(
            hotels=[], 
            ai_hotel_recommendation=f"An error occurred while processing hotel data: {str(e)}"
        )

@app.post("/generate_itinerary/", response_model=AIResponse)
async def get_itinerary(itinerary_request: ItineraryRequest):
    itinerary = await generate_itinerary(
        itinerary_request.destination,
        itinerary_request.flights,
        itinerary_request.hotels,
        itinerary_request.check_in_date,
        itinerary_request.check_out_date
    )
    return AIResponse(itinerary=itinerary)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run FastAPI Server
if __name__ == "__main__":
    logger.info("Starting Travel Planning API server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)